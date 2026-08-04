#!/usr/bin/env python3
"""world -> target_object_{coarse,fine}のTFを取得し、py_trees(py_trees_ros)によるbehavior
treeでアプローチ→下降→把持→上昇→プレイスの一連動作を実行するピッキングコントローラ。

俯瞰カメラでの粗検出(target_object_coarse)でプレアプローチ位置まで移動し、
そこで手先カメラでの精緻検出(target_object_fine)を待ってから下降する、という
2段階の姿勢推定に対応する。実際のツリー構築はbuild_tree()、各ステップの実処理は
picking_robot_interface.RobotInterfaceとpicking_behaviours.pyの各behaviourに委譲する。

moveit_py標準のtrajectory_execution(ros2_controlのFollowJointTrajectoryアクション)は
Isaac Sim側に存在せず、run_simulation.pyのOmniGraphは/joint_command(sensor_msgs/JointState)を
直接subscribeするだけの構成になっている。そのため、計画したtrajectoryのwaypointを
RobotInterfaceがタイマー無しの逐次sleepループで/joint_commandへpublishする、
簡易的なtrajectoryプレイヤーを実装している(この一連の処理は同期的にブロックするため、
本ツリーの各behaviourも一発実行・即座にSUCCESS/FAILUREを返す設計にしている)。
"""

import sys
import threading
import time

import py_trees
import py_trees_ros
import rclpy

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
            MoveTo(
                "下降",
                interface,
                height_offset=0.0,
                log_label="下降",
                position_key="fine_pose",
                orientation_key="fine_pose",
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

    try:
        while rclpy.ok() and root.status not in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            tree.tick()
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
