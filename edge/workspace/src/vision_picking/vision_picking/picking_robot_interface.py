#!/usr/bin/env python3
"""picking_controller_node.pyのbehaviour群が共有するロボット操作インターフェース。

moveit_py・TF・joint_command publisher・collision object操作など、実際にロボットへ
働きかける処理をまとめる。py_trees.behaviour.Behaviourのサブクラス(picking_behaviours.py)は
薄いラッパーとしてこのクラスのメソッドを呼ぶだけにし、ロジックの重複を避ける。

このクラスはrclpy.node.Nodeを継承せず、外部(py_trees_ros.trees.BehaviourTree.setup()が
内部生成する、またはmain()で明示的に渡すノード)から`node`を受け取る。behaviour側が
tickのたびに再購読・再生成しなくて済むよう、TF listener・publisher・subscriptionは
このクラスの初期化時に1度だけ作る。
"""

import math
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from common.target_object_shape import OBJECT_SIZE_M as TARGET_SIZE
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit.planning import MoveItPy
from moveit.utils import create_params_file_from_dict
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_msgs.msg import TFMessage
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion

GROUND_TRUTH_TF_TOPIC = "/ground_truth/tf"
# 把持が失敗していても_attach_target()はMoveItの計画上のアタッチを無条件に成功させてしまい、
# 実際にIsaac Sim側の物理シミュレーションで物体を運べたかどうかとは無関係に
# 「ピック&プレイス完了」まで到達してしまう。run_simulation.pyが常時配信する
# /ground_truth/tf(認識方式によらない物理的な真の位置)でプレイス後の実際の位置を検証し、
# ログ上の完了と物理的な成否を区別できるようにする。
GRASP_SUCCESS_XY_TOLERANCE_M = 0.05
# 上昇後、target_objectがこの高さ以上持ち上がっていなければ把持失敗とみなす
# (APPROACH_HEIGHT=0.10の半分。指の間からすり抜けて一度も持ち上がらないまま
# プレイスへ向かってしまう問題を、プレイス降下・リリースの前に検出するため)。
MIN_LIFT_HEIGHT_M = 0.05

WORLD_FRAME = "world"
COARSE_TARGET_FRAME = "target_object_coarse"
FINE_TARGET_FRAME = "target_object_fine"
ARM_GROUP = "panda_arm"
END_EFFECTOR_LINK = "panda_link8"
JOINT_COMMAND_TOPIC = "/joint_command"
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_OPEN_POSITIONS = [0.035, 0.035]
# target_objectの半径(TARGET_SIZE/2=0.025m)より深く閉じる指令にすると、位置制御の
# フィンガーが軽量なキューブを弾き飛ばして指の間から押し出し、何も挟まないまま
# 0まで閉じきってしまう(/joint_statesの実測値で接触・押し出しの挙動を確認済み)。
# 半径よりわずかに手前で止めることで、キューブを押し出さずに挟み込む力を残す。
GRIPPER_CLOSED_POSITIONS = [0.01, 0.01]

APPROACH_HEIGHT = 0.10
PLACE_POSITION = (0.3, 0.3, TARGET_SIZE / 2.0)

# Ref: edge/workspace/.pixi/envs/default/.../isaacsim/exts/isaacsim.robot.experimental.manipulators.examples/
#      isaacsim/robot/experimental/manipulators/examples/franka/franka.py Franka.get_downward_orientation()
# 同メソッドが返す(w,x,y,z)=[0,1,0,0]をgeometry_msgs/Quaternion(x,y,z,w)の並びに変換したもの。
# panda_link8のローカルZ軸をworldの-Z(真下)へ向ける姿勢。
DOWNWARD_ORIENTATION = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

# panda_link8(フランジ)からフィンガー先端(把持点)までのオフセット。Frankaハンドの
# 標準的な長さ(約0.1034m)による近似値。
FLANGE_TO_FINGERTIP_Z = 0.1034


def grasp_orientation_for_yaw(rotation: Quaternion) -> Quaternion:
    # DOWNWARD_ORIENTATION固定のまま把持すると、target_objectが回転している場合に
    # 指がcubeの面ではなく角や斜めの位置に当たってしまい、掴み損ねる。
    # target_objectは正方形の断面を持つcubeで90度ごとに同じ形状が繰り返されるため、
    # yawを90度周期で畳み込んで最小回転で面に軸を合わせる。
    # panda_hand_joint(panda.urdf)がEND_EFFECTOR_LINK(panda_link8)に対して
    # rpy z=-0.785398(-45度)の固定回転を持つため、畳み込みの位相を45度分ずらして
    # 実際の指の開閉軸(panda_hand基準)がcubeの面に対して直角に合うようにしている。
    _, _, yaw = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
    yaw_mod = (yaw % (math.pi / 2)) - math.pi / 4
    return Quaternion(x=math.cos(yaw_mod / 2.0), y=math.sin(yaw_mod / 2.0), z=0.0, w=0.0)


class RobotInterface:
    def __init__(self, node: Node) -> None:
        self.node = node
        self._joint_command_pub = node.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._ground_truth_target_pose: Pose | None = None
        node.create_subscription(TFMessage, GROUND_TRUTH_TF_TOPIC, self._on_ground_truth_tf, 10)

        # Ref: https://github.com/moveit/moveit2/issues/2409#issuecomment-1753090620
        # moveit_resources_panda_moveit_configにはmoveit_cpp.yamlが同梱されておらず、
        # .moveit_cpp(file_path=...)を明示的に呼ばない限りMoveItPyがplanning_pipelinesを
        # 解決できずに初期化失敗する(Failed to load any planning pipelines)ため、
        # 本パッケージ側で用意したmoveit_cpp.yamlを明示的に読み込む。
        moveit_cpp_yaml_path = get_package_share_directory("vision_picking") + "/config/moveit_cpp.yaml"
        moveit_config = (
            MoveItConfigsBuilder("panda", package_name="moveit_resources_panda_moveit_config")
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
            .planning_scene_monitor(
                publish_robot_description=True,
                publish_robot_description_semantic=True,
            )
            .moveit_cpp(file_path=moveit_cpp_yaml_path)
            .to_moveit_configs()
        )
        # Ref: https://github.com/moveit/moveit2/issues/2940#issuecomment-2401302214
        # run_simulation.pyのOmniGraphは/joint_statesをシミュレーション時刻でタイムスタンプしており、
        # use_sim_timeを設定しないとMoveItPyの内部current_state_monitorが「最新の関節状態が古すぎる」
        # と判定してplanning scene monitorの初期化に失敗する。
        # Ref: https://github.com/moveit/moveit2/issues/2940#issuecomment-2464301172
        # ただしuse_sim_timeをconfig_dict経由でMoveItPyへ直接渡すと、内部ノードの
        # qos_overrides./clock.subscription.durabilityパラメータ設定でrclcpp例外が出て
        # 起動時にクラッシュする(moveit_py側の既知の問題)。config_dictをそのまま渡さず、
        # 一時YAMLパラメータファイルとして書き出しlaunch_params_filepaths経由で渡すことで回避する。
        config_dict = moveit_config.to_dict()
        config_dict["use_sim_time"] = True
        moveit_params_file = create_params_file_from_dict(config_dict, "/**")
        self.moveit_py = MoveItPy(
            node_name="picking_controller_moveit_py",
            launch_params_filepaths=[moveit_params_file],
        )
        self._arm = self.moveit_py.get_planning_component(ARM_GROUP)
        self._psm = self.moveit_py.get_planning_scene_monitor()

    def _on_ground_truth_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id == "target_object":
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation = transform.transform.rotation
                self._ground_truth_target_pose = pose

    def verify_grasped(self, pick_z: float) -> None:
        if self._ground_truth_target_pose is None:
            raise RuntimeError(f"{GROUND_TRUTH_TF_TOPIC}を一度も受信できておらず、把持できたか検証できません")
        z = self._ground_truth_target_pose.position.z
        if z < pick_z + MIN_LIFT_HEIGHT_M:
            raise RuntimeError(
                f"把持失敗と判定(上昇後もtarget_objectが持ち上がっていません): "
                f"実際の高さ={z:.3f}m、必要な高さ={pick_z + MIN_LIFT_HEIGHT_M:.3f}m以上"
            )
        self.node.get_logger().info(f"把持確認OK: target_objectの高さ={z:.3f}mまで持ち上がっています")

    def verify_placed(self) -> None:
        place_x, place_y, _ = PLACE_POSITION
        if self._ground_truth_target_pose is None:
            raise RuntimeError(f"{GROUND_TRUTH_TF_TOPIC}を一度も受信できておらず、実際に置けたか検証できません")
        x = self._ground_truth_target_pose.position.x
        y = self._ground_truth_target_pose.position.y
        z = self._ground_truth_target_pose.position.z
        xy_error = math.hypot(x - place_x, y - place_y)
        if xy_error > GRASP_SUCCESS_XY_TOLERANCE_M:
            raise RuntimeError(
                f"ピック失敗と判定(target_objectを実際には運べていません): "
                f"実際の最終位置=({x:.3f}, {y:.3f}, {z:.3f})、"
                f"プレイス目標=({place_x:.3f}, {place_y:.3f})からの距離={xy_error:.3f}m"
            )
        self.node.get_logger().info(f"検証OK: target_objectは実際にプレイス目標から{xy_error:.3f}m以内の位置にあります")

    def log_finger_positions(self, label: str, target_xyz: tuple[float, float, float]) -> None:
        with self._psm.read_only() as scene:
            state = scene.current_state
            for link in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
                pose = state.get_pose(link)
                p = pose.position
                self.node.get_logger().info(
                    f"[DEBUG {label}] {link}: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
                    f"target=({target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f})  "
                    f"dz={p.z - target_xyz[2]:.4f}"
                )

    def lookup_target_pose(self, target_frame: str) -> Pose | None:
        # BT側(LookupTargetPose behaviour)がtickのたびに1回だけ試行し、
        # 見つからなければRUNNINGを返して次のtickで再試行する設計にするため、
        # ここではブロッキングせず1回の試行のみ行う。
        try:
            transform = self._tf_buffer.lookup_transform(WORLD_FRAME, target_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = grasp_orientation_for_yaw(transform.transform.rotation)
        return pose

    def register_collision_object(self, frame_id: str, object_id: str, pose: Pose) -> None:
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = frame_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [TARGET_SIZE, TARGET_SIZE, TARGET_SIZE]
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        self._psm.process_collision_object(collision_object_msg=obj)

    def remove_collision_object(self, object_id: str) -> None:
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = WORLD_FRAME
        obj.operation = CollisionObject.REMOVE
        self._psm.process_collision_object(collision_object_msg=obj)

    def move_to(
        self, position_xyz: tuple[float, float, float], orientation: Quaternion = DOWNWARD_ORIENTATION
    ) -> None:
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = WORLD_FRAME
        pose_goal.pose.position.x = position_xyz[0]
        pose_goal.pose.position.y = position_xyz[1]
        pose_goal.pose.position.z = position_xyz[2] + FLANGE_TO_FINGERTIP_Z
        pose_goal.pose.orientation = orientation

        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=END_EFFECTOR_LINK)
        plan_result = self._arm.plan()
        if not plan_result:
            raise RuntimeError(f"軌道計画に失敗しました: target={position_xyz}")
        self._play_trajectory(plan_result.trajectory)

    def _play_trajectory(self, robot_trajectory) -> None:
        joint_trajectory = robot_trajectory.get_robot_trajectory_msg().joint_trajectory
        joint_names = list(joint_trajectory.joint_names)
        start_time = time.monotonic()
        for point in joint_trajectory.points:
            target_elapsed = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            remaining = target_elapsed - (time.monotonic() - start_time)
            if remaining > 0:
                time.sleep(remaining)
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = list(point.positions)
            self._joint_command_pub.publish(msg)

    def set_gripper(self, positions: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = GRIPPER_JOINT_NAMES
        msg.position = positions
        self._joint_command_pub.publish(msg)
        time.sleep(2.0)

    def attach_target(self) -> None:
        # 下降前にプランニングシーンからtarget_objectを外している(remove_collision_object)ため、
        # ジオメトリを持たないAttachedCollisionObjectをADDしても中身が空になってしまう。
        # end-effectorからのローカル姿勢込みでジオメトリを再定義してアタッチする。
        attached = AttachedCollisionObject()
        attached.link_name = END_EFFECTOR_LINK
        attached.object.id = "target_object"
        attached.object.header.frame_id = END_EFFECTOR_LINK
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [TARGET_SIZE, TARGET_SIZE, TARGET_SIZE]
        attached.object.primitives.append(primitive)
        local_pose = Pose()
        local_pose.position.z = FLANGE_TO_FINGERTIP_Z
        local_pose.orientation.w = 1.0
        attached.object.primitive_poses.append(local_pose)
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        self._psm.process_attached_collision_object(attached_collision_object_msg=attached)

    def detach_target(self) -> None:
        attached = AttachedCollisionObject()
        attached.link_name = END_EFFECTOR_LINK
        attached.object.id = "target_object"
        attached.object.operation = CollisionObject.REMOVE
        self._psm.process_attached_collision_object(attached_collision_object_msg=attached)
