#!/usr/bin/env python3
"""world -> target_object_{coarse,fine}のTFを取得し、py_trees(py_trees_ros)によるbehavior
treeでアプローチ→下降→把持→上昇→プレイスの一連動作を実行するピッキングコントローラ。

俯瞰カメラでの粗検出(target_object_coarse)でプレアプローチ位置まで移動し、
そこで手先カメラでの精緻検出(target_object_fine)を待ってから下降する、という
2段階の姿勢推定に対応する。実際のツリー構築はbuild_tree()、各ステップの実処理は
picking_robot_interface.RobotInterfaceとpicking_behaviours.pyの各behaviourに委譲する。

アーム・グリッパーの実行はros2_control(controller_manager、要:別途vp-controller-manager起動)経由の
FollowJointTrajectory/GripperCommandアクションを使い、moveit_py標準のexecute()で行う
(picking_robot_interface.RobotInterface参照)。この一連の処理は完了まで同期的にブロックするため、
本ツリーの各behaviourも一発実行・即座にSUCCESS/FAILUREを返す設計にしている。
"""

import sys
import threading
import time

import py_trees
import py_trees_ros
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

from vision_picking.diagnostics import DIAGNOSTICS_TOPIC, build_diagnostics
from vision_picking.picking_behaviours import (
    AttachTarget,
    DetachTarget,
    LogFingerPositions,
    LookupTargetPose,
    MoveTo,
    RegisterCollisionObject,
    RemoveCollisionObject,
    SetGripper,
    VerifyGrasped,
    VerifyPlaced,
)
from vision_picking.picking_robot_interface import (
    APPROACH_HEIGHT,
    COARSE_TARGET_FRAME,
    FINE_TARGET_FRAME,
    GRIPPER_CLOSED_POSITIONS,
    GRIPPER_OPEN_POSITIONS,
    PLACE_POSITION,
    RobotInterface,
)

TICK_PERIOD_SEC = 0.1
TREE_SETUP_TIMEOUT_SEC = 15.0

_DIAGNOSTIC_LEVEL = {
    py_trees.common.Status.SUCCESS: DiagnosticStatus.OK,
    py_trees.common.Status.RUNNING: DiagnosticStatus.WARN,
    py_trees.common.Status.FAILURE: DiagnosticStatus.ERROR,
    py_trees.common.Status.INVALID: DiagnosticStatus.STALE,
}


def _publish_diagnostics(
    node: rclpy.node.Node, publisher, root: py_trees.behaviour.Behaviour
) -> None:
    statuses = [
        ("picking_controller: 全体", _DIAGNOSTIC_LEVEL.get(root.status, DiagnosticStatus.STALE), root.status.name)
    ]
    # Sequence自体(root)の状態はoverallで表現済みのため、composite自身は除外し
    # 葉のbehaviourだけを個別のステップとして列挙する。
    for behaviour in root.iterate():
        if isinstance(behaviour, py_trees.composites.Composite):
            continue
        statuses.append((
            f"picking_controller: {behaviour.name}",
            _DIAGNOSTIC_LEVEL.get(behaviour.status, DiagnosticStatus.STALE),
            behaviour.status.name,
        ))
    publisher.publish(build_diagnostics(node, "picking_controller_node", statuses))


def build_tree(interface: RobotInterface) -> py_trees.behaviour.Behaviour:
    return py_trees.composites.Sequence(
        name="pick_and_place",
        memory=True,
        children=[
            LookupTargetPose(
                "粗検出待ち", interface, target_frame=COARSE_TARGET_FRAME, blackboard_key="coarse_pose"
            ),
            RegisterCollisionObject("coarse姿勢で衝突オブジェクト登録", interface, position_key="coarse_pose"),
            MoveTo(
                "プレアプローチ",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="プレアプローチ",
                position_key="coarse_pose",
                orientation_key="coarse_pose",
            ),
            SetGripper("グリッパーオープン", interface, positions=GRIPPER_OPEN_POSITIONS),
            LookupTargetPose(
                "精緻検出待ち", interface, target_frame=FINE_TARGET_FRAME, blackboard_key="fine_pose"
            ),
            RemoveCollisionObject("下降前に衝突オブジェクト除去", interface),
            # coarse(俯瞰)とfine(手先)の推定yawはズレうるため、プレアプローチ高さのまま
            # fine姿勢へ合わせ直す。1回のMoveToで回転・水平移動・下降をまとめて行うと、
            # OMPLは各関節を同時に動かす軌道を返すため回転と並進が混ざり、指が物体に
            # 近い高さで回転しながら動くことになって物体や台にひっかかりやすい。
            # 「その場で回転(位置はcoarse姿勢のまま)」→「水平移動(向きはfineのまま)」→
            # 「垂直下降(向き・水平位置ともに変えない)」の3段階に分け、各段階で
            # 変化させるのを位置か向きのどちらか一方だけにする。
            MoveTo(
                "その場で姿勢合わせ",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="その場で姿勢合わせ",
                position_key="coarse_pose",
                orientation_key="fine_pose",
                yaw_free=True,
            ),
            MoveTo(
                "精緻位置へ水平移動",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="精緻位置へ水平移動",
                position_key="fine_pose",
                orientation_key="fine_pose",
                yaw_free=True,
            ),
            MoveTo(
                "下降",
                interface,
                height_offset=0.0,
                log_label="下降",
                position_key="fine_pose",
                orientation_key="fine_pose",
                verify_and_correct=True,
                yaw_free=True,
            ),
            LogFingerPositions("下降後ログ", interface, label="下降後", position_key="fine_pose"),
            SetGripper("把持", interface, positions=GRIPPER_CLOSED_POSITIONS, log_label="把持"),
            AttachTarget("アタッチ", interface),
            LogFingerPositions("把持後ログ", interface, label="把持後", position_key="fine_pose"),
            MoveTo(
                "上昇",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="上昇",
                position_key="fine_pose",
                orientation_key="fine_pose",
            ),
            VerifyGrasped("把持確認", interface, position_key="fine_pose"),
            MoveTo(
                "プレイス位置上空へ移動",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="プレイス位置上空へ移動",
                static_position=PLACE_POSITION,
                orientation_key="fine_pose",
            ),
            MoveTo(
                "プレイス降下",
                interface,
                height_offset=0.0,
                log_label="プレイス降下",
                static_position=PLACE_POSITION,
                orientation_key="fine_pose",
            ),
            SetGripper("リリース", interface, positions=GRIPPER_OPEN_POSITIONS, log_label="リリース"),
            DetachTarget("デタッチ", interface),
            RemoveCollisionObject("リリース後に衝突オブジェクト除去", interface),
            # 指がまだ物体のすぐ脇にある位置で衝突オブジェクトを再登録すると、指の開き幅と
            # 物体サイズの余裕がほぼ無くCheckStartStateCollisionに失敗するため、
            # 退避(向きはDOWNWARD_ORIENTATIONに戻す。物体を保持していないため姿勢を
            # 維持する必要が無い)で十分離れてから再登録する。
            MoveTo(
                "退避",
                interface,
                height_offset=APPROACH_HEIGHT,
                log_label="退避",
                static_position=PLACE_POSITION,
            ),
            RegisterCollisionObject(
                "プレイス後に衝突オブジェクト再登録",
                interface,
                static_position=PLACE_POSITION,
                orientation_key="fine_pose",
            ),
            VerifyPlaced("検証", interface),
        ],
    )


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("picking_controller_node")
    interface = RobotInterface(node)

    # tree.setup()がnode上にサービス・パラメータを追加するため、RobotInterfaceの初期化
    # (TF listener・moveit_py等)が終わった後にスピンを開始する。
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = build_tree(interface)
    tree = py_trees_ros.trees.BehaviourTree(root=root)
    tree.setup(node=node, timeout=TREE_SETUP_TIMEOUT_SEC)
    diagnostics_pub = node.create_publisher(DiagnosticArray, DIAGNOSTICS_TOPIC, 10)

    try:
        while rclpy.ok() and root.status not in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            tree.tick()
            _publish_diagnostics(node, diagnostics_pub, root)
            time.sleep(TICK_PERIOD_SEC)
    finally:
        # spin_threadがexecutorをスピンしたままmoveit_py.shutdown()/destroy_node()を呼ぶと、
        # 実行中のコールバックとリソース解放が競合してabortする。rclpy.shutdown()を先に呼んで
        # spin()側にExternalShutdownExceptionを送出させ、スレッド終了(join)を待ってから
        # 後片付けする。
        final_status = root.status
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
        interface.moveit_py.shutdown()
        node.destroy_node()

    if final_status != py_trees.common.Status.SUCCESS:
        sys.exit(1)


if __name__ == "__main__":
    main()
