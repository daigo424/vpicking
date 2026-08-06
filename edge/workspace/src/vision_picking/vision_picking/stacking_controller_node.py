#!/usr/bin/env python3
"""ランダム配置された複数のtarget_objectを、俯瞰/手先カメラのYOLO認識(vp-run-yoloと同じ仕組み)で
1個ずつ検出しながら順番に掴み、指定位置に積み重ねる(スタッキング)コントローラ。

picking_controller_node.pyの単一物体ピック&プレイスとは別の、独立したノードとして実装する。
YOLOモデル自体は「キューブらしきもの」を1個検出するだけで個体識別はしないため、
複数個が卓上にあっても既存のpose_estimation_node.py側は無改修のまま、毎回
「その時点で最も検出しやすい1個」を素直に拾わせる形で動く。掴んだ後は卓上から
運び去られるため、次の周回では自然に残りの物体が検出対象になる。

RobotInterface(移動・グリッパー・衝突オブジェクト操作)はpicking_controller_nodeと共通で
再利用し、ロジックの重複を避ける。一方、検証(掴めたか・積めたか)は複数物体の個体識別が
必要になるため、picking_controller_node側のverify_grasped/verify_placed(単一物体・
固定プレイス位置前提)とは別に、ここではground truthの複数物体トラッキングを
このノード内に閉じて実装する。

事前にsim(STACK_BLOCKS環境変数で複数物体を生成)・camera_bridge_node・
pose_estimation_node(俯瞰=coarse・手先=fine)を別途起動しておく必要がある。
"""

import math
import sys
import threading
import time

import rclpy
from common.target_object_shape import OBJECT_SIZE_M as TARGET_SIZE
from common.target_object_shape import TABLE_HEIGHT_M
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Pose
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from vision_picking.diagnostics import DIAGNOSTICS_TOPIC, build_diagnostics
from vision_picking.picking_robot_interface import (
    APPROACH_HEIGHT,
    COARSE_TARGET_FRAME,
    DOWNWARD_ORIENTATION,
    FINE_TARGET_FRAME,
    GRIPPER_CLOSED_POSITIONS,
    GRIPPER_OPEN_POSITIONS,
    RobotInterface,
    WORLD_FRAME,
)

GROUND_TRUTH_TF_TOPIC = "/ground_truth/tf"
BLOCK_FRAME_PREFIX = "stack_block_"
NUM_BLOCKS = 3
STACK_BASE_XY = (0.3, 0.3)
POSE_LOOKUP_TIMEOUT_SEC = 30.0
# 上昇後、積んだ後の判定を持ち上がった/積めたとみなす許容値(picking_robot_interfaceの
# 単一物体版と同じ値を使う)。
MIN_LIFT_HEIGHT_M = 0.05
PLACE_XY_TOLERANCE_M = 0.05
PLACE_Z_TOLERANCE_M = 0.02
# 俯瞰カメラの画角を広げたことで積み台がはっきり画角に入るようになり、個体識別のない
# 検出モデルが未配置のブロックではなく既に積んだブロックを最有力候補として検出することが
# ある。ground truthとの名前ベースの除外(exclude=placed_blocks)だけでは、検出座標自体が
# 積み台に張り付いている場合を防げないため、検出座標と既に積んだブロックの実位置との
# 距離でも別途弾く。
COARSE_PLACED_EXCLUSION_RADIUS_M = 0.15

# move_to()は計画軌道を時刻通りに再生するだけで、シミュレーター側のロボットが実際に
# 最終姿勢へ到達したかは見ていない。障害物回避で複雑になった軌道では追従遅れが起きやすく、
# 到達前にリリースすると物体を狙った位置からずれた場所で落としてしまう。保持中の物体の
# ground truth位置が数フレーム安定するまで待ってからリリースする。
SETTLE_WINDOW = 5
SETTLE_TOLERANCE_M = 0.002
SETTLE_TIMEOUT_SEC = 3.0

# 実際のキューブは無彩色のまま(検出モデルへ渡す画像を色付きにすると検出が不安定になるため)、
# Foxglove側だけで見分けられるよう、各物体のground truth位置に3Dマーカーを重ねて色分けする。
BLOCK_MARKER_COLORS = [
    ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.6),
    ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.6),
    ColorRGBA(r=0.2, g=0.2, b=0.8, a=0.6),
]
BLOCK_MARKERS_TOPIC = "/stacking/block_markers"


def _publish_diagnostics(
    node,
    publisher,
    layer_levels: list[int],
    layer_messages: list[str],
    overall_level: int,
    overall_message: str,
) -> None:
    statuses = [("stacking_controller: 全体", overall_level, overall_message)]
    for i, (level, message) in enumerate(zip(layer_levels, layer_messages)):
        statuses.append((f"stacking_controller: {i + 1}段目", level, message))
    publisher.publish(build_diagnostics(node, "stacking_controller_node", statuses))


class GroundTruthTracker:
    """world系でのstack_block_<N>各物体の最新姿勢をトラッキングする。"""

    def __init__(self, node) -> None:
        self._poses: dict[str, Pose] = {}
        self._marker_pub = node.create_publisher(MarkerArray, BLOCK_MARKERS_TOPIC, 10)
        # Foxgloveはlifetimeが切れるかDELETEATTを受け取るまでマーカーを表示し続けるため、
        # 前回セッションの残留マーカー(例: 積み終わった状態)がFoxglove上に残ったまま、
        # 実際のsim側の状態(まだ何も積んでいない等)と食い違って見えることがある。
        # 起動時に明示的に消去しておく。
        clear = MarkerArray()
        clear_marker = Marker()
        clear_marker.ns = "stack_blocks"
        clear_marker.action = Marker.DELETEALL
        clear.markers.append(clear_marker)
        self._marker_pub.publish(clear)
        node.create_subscription(TFMessage, GROUND_TRUTH_TF_TOPIC, self._on_tf, 10)

    def _on_tf(self, msg: TFMessage) -> None:
        updated = False
        for transform in msg.transforms:
            if not transform.child_frame_id.startswith(BLOCK_FRAME_PREFIX):
                continue
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            self._poses[transform.child_frame_id] = pose
            updated = True
        if updated:
            self._publish_markers()

    def _publish_markers(self) -> None:
        markers = MarkerArray()
        for name, pose in sorted(self._poses.items()):
            index = int(name[len(BLOCK_FRAME_PREFIX) :]) - 1
            marker = Marker()
            marker.header.frame_id = WORLD_FRAME
            marker.ns = "stack_blocks"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = marker.scale.y = marker.scale.z = TARGET_SIZE
            marker.color = BLOCK_MARKER_COLORS[index % len(BLOCK_MARKER_COLORS)]
            # /ground_truth/tfは常時高頻度(数十Hz)で更新され続けるため、更新が数秒止まる
            # (このノードが終了した等)場合だけFoxglove側で自動的に消えるようにしておく。
            marker.lifetime.sec = 3
            markers.markers.append(marker)
        self._marker_pub.publish(markers)

    def has_any_pose(self) -> bool:
        return bool(self._poses)

    def closest_block(self, x: float, y: float, exclude: set[str] = frozenset()) -> tuple[str, Pose] | None:
        candidates = {name: pose for name, pose in self._poses.items() if name not in exclude}
        if not candidates:
            return None
        name = min(candidates, key=lambda n: math.hypot(candidates[n].position.x - x, candidates[n].position.y - y))
        return name, candidates[name]

    def pose_of(self, name: str) -> Pose | None:
        return self._poses.get(name)


def _wait_until_settled(tracker: GroundTruthTracker, block_name: str, node) -> None:
    # ポーリング間隔をground truthの配信周期(30〜45Hz)より広く取り、同じサンプルを
    # 重複して読んで安定と誤判定しないようにする。
    deadline = time.monotonic() + SETTLE_TIMEOUT_SEC
    recent: list[Pose] = []
    while time.monotonic() < deadline:
        pose = tracker.pose_of(block_name)
        if pose is not None:
            recent.append(pose)
            if len(recent) > SETTLE_WINDOW:
                recent.pop(0)
            if len(recent) == SETTLE_WINDOW:
                spread = max(
                    math.hypot(a.position.x - b.position.x, a.position.y - b.position.y) + abs(a.position.z - b.position.z)
                    for i, a in enumerate(recent)
                    for b in recent[i + 1 :]
                )
                if spread < SETTLE_TOLERANCE_M:
                    return
        time.sleep(0.1)
    node.get_logger().info(f"{block_name}の静止確認がタイムアウトしたため、そのままリリースします")


def _wait_for_pose(interface: RobotInterface, target_frame: str, node) -> Pose:
    deadline = time.monotonic() + POSE_LOOKUP_TIMEOUT_SEC
    logged = False
    while time.monotonic() < deadline:
        pose = interface.lookup_target_pose(target_frame)
        if pose is not None:
            return pose
        if not logged:
            node.get_logger().info(f"{target_frame}のTFを待機中...")
            logged = True
        time.sleep(0.1)
    raise RuntimeError(f"{target_frame}のTFがタイムアウトまでに取得できませんでした")


def _wait_for_unplaced_pose(
    interface: RobotInterface, tracker: GroundTruthTracker, node, placed_blocks: set[str], target_frame: str
) -> Pose:
    # 個体識別のない検出モデルは、俯瞰(coarse)だけでなく手先(fine)側でも既に積んだブロックを
    # 誤って検出しうるため、両方のTF取得でこのチェックを共有する。
    deadline = time.monotonic() + POSE_LOOKUP_TIMEOUT_SEC
    warned = False
    while time.monotonic() < deadline:
        pose = interface.lookup_target_pose(target_frame)
        if pose is not None:
            placed_positions = [tracker.pose_of(name) for name in placed_blocks]
            near_placed = any(
                p is not None
                and math.hypot(pose.position.x - p.position.x, pose.position.y - p.position.y)
                < COARSE_PLACED_EXCLUSION_RADIUS_M
                for p in placed_positions
            )
            if not near_placed:
                return pose
            if not warned:
                node.get_logger().info(f"{target_frame}が積み台(既に積んだブロック)付近を捉えているため、再検出を待機中...")
                warned = True
        time.sleep(0.1)
    raise RuntimeError(f"{target_frame}の有効なTFがタイムアウトまでに取得できませんでした")


def pick_and_stack_one(
    interface: RobotInterface,
    tracker: GroundTruthTracker,
    node,
    layer: int,
    placed_blocks: set[str],
    place_x: float,
    place_y: float,
    place_z: float,
) -> tuple[str, Pose]:
    coarse_pose = _wait_for_unplaced_pose(interface, tracker, node, placed_blocks, COARSE_TARGET_FRAME)
    # 検出座標自体は積み台から十分離れていることを上で確認済みだが、それでも万一
    # 積み終わった物体をground truthと誤って対応付けないよう、名前ベースでも除外する。
    pre_pick = tracker.closest_block(coarse_pose.position.x, coarse_pose.position.y, exclude=placed_blocks)
    if pre_pick is None:
        if not tracker.has_any_pose():
            raise RuntimeError(
                f"{BLOCK_FRAME_PREFIX}*のground truthを一度も受信できていません。"
                "simをSTACK_BLOCKS環境変数付き(make sim-stack)で起動しているか確認してください"
            )
        raise RuntimeError("未配置のブロックがground truth上に見つかりません(全て積み終わっている可能性があります)")
    block_name, pre_pick_pose = pre_pick
    pick_z = pre_pick_pose.position.z
    node.get_logger().info(f"{layer + 1}個目: {block_name}を検出(粗)")

    interface.register_collision_object(WORLD_FRAME, "target_object", coarse_pose)
    interface.move_to((coarse_pose.position.x, coarse_pose.position.y, coarse_pose.position.z + APPROACH_HEIGHT), coarse_pose.orientation)
    interface.set_gripper(GRIPPER_OPEN_POSITIONS)

    fine_pose = _wait_for_unplaced_pose(interface, tracker, node, placed_blocks, FINE_TARGET_FRAME)
    interface.remove_collision_object("target_object")

    # coarse/fineのyaw推定がズレうるため、下降と回転を同時に行わず
    # 「その場で姿勢合わせ」→「精緻位置へ水平移動」→「下降」の3段階に分ける
    # (picking_controller_node.pyの下降ステップと同じ理由による分割)。
    interface.move_to(
        (coarse_pose.position.x, coarse_pose.position.y, fine_pose.position.z + APPROACH_HEIGHT), fine_pose.orientation
    )
    interface.move_to((fine_pose.position.x, fine_pose.position.y, fine_pose.position.z + APPROACH_HEIGHT), fine_pose.orientation)
    interface.move_to(
        (fine_pose.position.x, fine_pose.position.y, fine_pose.position.z), fine_pose.orientation, yaw_free=True
    )

    interface.set_gripper(GRIPPER_CLOSED_POSITIONS)
    interface.attach_target()
    # attach_target()直後にmove_to()を呼ぶと、touch_linksがPlanning Scene Monitorへ
    # まだ反映されておらず、指とtarget_object(アタッチ直後の自分自身)の接触が
    # 衝突として誤検出されCheckStartStateCollisionで計画が拒否されることがある
    # (time.sleepだけでは解消しなかった)。log_finger_positions()が内部で
    # self._psm.read_only()を経由してplanning sceneを読むことで同期を促せるため、
    # picking_controller_node.py(AttachTarget直後にLogFingerPositionsを挟む)と
    # 同じ順序にする。
    interface.log_finger_positions("アタッチ後", (fine_pose.position.x, fine_pose.position.y, fine_pose.position.z))
    interface.move_to((fine_pose.position.x, fine_pose.position.y, fine_pose.position.z + APPROACH_HEIGHT), fine_pose.orientation)

    lifted_pose = tracker.pose_of(block_name)
    if lifted_pose is None or lifted_pose.position.z < pick_z + MIN_LIFT_HEIGHT_M:
        raise RuntimeError(f"{block_name}の把持に失敗しました(持ち上がっていません)")
    node.get_logger().info(f"{block_name}の把持確認OK")

    # 掴んだ時の向き(fine_pose.orientation、物体ごとに元のランダムなyawに依存する)のまま置くと、
    # 積んだ物体同士の角度が揃わない。物体を保持したまま(位置は変えず)固定の基準向きへ
    # その場で回転させてから運ぶことで、どの物体を掴んでも積み上がった見た目を揃える。
    interface.move_to((fine_pose.position.x, fine_pose.position.y, fine_pose.position.z + APPROACH_HEIGHT), DOWNWARD_ORIENTATION)
    interface.move_to((place_x, place_y, place_z + APPROACH_HEIGHT), DOWNWARD_ORIENTATION)
    interface.move_to((place_x, place_y, place_z), DOWNWARD_ORIENTATION)
    _wait_until_settled(tracker, block_name, node)
    interface.set_gripper(GRIPPER_OPEN_POSITIONS)
    interface.detach_target()
    # detach_target()はアタッチを解除するだけで、リリースされた物体はその場でワールド座標系の
    # 通常の衝突オブジェクト"target_object"として残る。退避で指がすぐ脇を通るとこれに
    # 衝突判定で引っかかるため、picking_controller_node.pyと同様に明示的に除去する。
    interface.remove_collision_object("target_object")
    interface.log_finger_positions("デタッチ後", (place_x, place_y, place_z))
    interface.move_to((place_x, place_y, place_z + APPROACH_HEIGHT), DOWNWARD_ORIENTATION)

    # 積んだ物体はこの後の周回でも衝突対象として残す(idをlayerごとに分け、
    # 次の物体をつかむ際のtarget_objectの登録・除去と競合しないようにする)。
    placed_pose = Pose()
    placed_pose.position.x, placed_pose.position.y, placed_pose.position.z = place_x, place_y, place_z
    placed_pose.orientation = DOWNWARD_ORIENTATION
    interface.register_collision_object(WORLD_FRAME, f"stack_layer_{layer}", placed_pose)

    placed = tracker.pose_of(block_name)
    if placed is None:
        raise RuntimeError(f"{block_name}の最終位置が確認できません")
    xy_error = math.hypot(placed.position.x - place_x, placed.position.y - place_y)
    if xy_error > PLACE_XY_TOLERANCE_M or abs(placed.position.z - place_z) > PLACE_Z_TOLERANCE_M:
        raise RuntimeError(
            f"{block_name}の積み上げに失敗と判定: 実際の位置=({placed.position.x:.3f}, {placed.position.y:.3f}, "
            f"{placed.position.z:.3f})、目標=({place_x:.3f}, {place_y:.3f}, {place_z:.3f})"
        )
    node.get_logger().info(f"{block_name}を{layer + 1}段目に積みました(XY誤差{xy_error:.3f}m)")
    return block_name, placed


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("stacking_controller_node")
    interface = RobotInterface(node)
    tracker = GroundTruthTracker(node)
    diagnostics_pub = node.create_publisher(DiagnosticArray, DIAGNOSTICS_TOPIC, 10)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # pick_and_stack_one()は1段分をまとめて同期的にブロックする粒度でしか実行できないため、
    # picking_controller_node.pyのようなtick単位の詳細な状態は出せない。
    # 「何段目が処理中/完了/失敗か」だけを段単位で配信する。
    layer_levels = [DiagnosticStatus.STALE] * NUM_BLOCKS
    layer_messages = ["未着手"] * NUM_BLOCKS

    success = True
    try:
        placed_blocks: set[str] = set()
        # 常に固定の目標位置(STACK_BASE_XY)を狙うと、前段の実際の着地位置とのズレが
        # 積み重なり、段を追うごとに着地面が偏って崩れやすくなる。1段目以降は
        # 前段の実測(ground truth)位置を次段の目標にすることで、ズレの蓄積を防ぐ。
        place_x, place_y = STACK_BASE_XY
        place_z = TABLE_HEIGHT_M + TARGET_SIZE * 0.5
        for layer in range(NUM_BLOCKS):
            layer_levels[layer], layer_messages[layer] = DiagnosticStatus.WARN, "処理中"
            _publish_diagnostics(
                node, diagnostics_pub, layer_levels, layer_messages, DiagnosticStatus.WARN, f"{layer + 1}段目を処理中"
            )
            block_name, placed_pose = pick_and_stack_one(
                interface, tracker, node, layer, placed_blocks, place_x, place_y, place_z
            )
            layer_levels[layer], layer_messages[layer] = DiagnosticStatus.OK, f"{block_name}を積みました"
            _publish_diagnostics(
                node, diagnostics_pub, layer_levels, layer_messages, DiagnosticStatus.OK, f"{layer + 1}段目完了"
            )
            placed_blocks.add(block_name)
            place_x, place_y = placed_pose.position.x, placed_pose.position.y
            place_z = placed_pose.position.z + TARGET_SIZE
        _publish_diagnostics(
            node, diagnostics_pub, layer_levels, layer_messages, DiagnosticStatus.OK, "3個のスタッキングが完了しました"
        )
        node.get_logger().info("3個のスタッキングが完了しました")
    except RuntimeError as e:
        layer_levels[layer], layer_messages[layer] = DiagnosticStatus.ERROR, str(e)
        _publish_diagnostics(node, diagnostics_pub, layer_levels, layer_messages, DiagnosticStatus.ERROR, str(e))
        node.get_logger().error(str(e))
        success = False
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
        interface.moveit_py.shutdown()
        node.destroy_node()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
