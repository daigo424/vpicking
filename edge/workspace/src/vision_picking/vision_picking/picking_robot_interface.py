#!/usr/bin/env python3
"""picking_controller_node.pyのbehaviour群が共有するロボット操作インターフェース。

moveit_py・TF・アーム/グリッパーの軌道実行・collision object操作など、実際にロボットへ
働きかける処理をまとめる。py_trees.behaviour.Behaviourのサブクラス(picking_behaviours.py)は
薄いラッパーとしてこのクラスのメソッドを呼ぶだけにし、ロジックの重複を避ける。

アーム・グリッパーの実行はros2_control(controller_manager、要:別途vp-controller-manager起動)が
提供するFollowJointTrajectory/GripperCommandアクション経由で行う。moveit_py標準の
execute()を使うため、計画された軌道の各時刻に本当に到達したかをmoveit_py/controller_manager側の
実行完了フィードバックで確認できる(タイマーで時刻通りに関節コマンドをpublishするだけの
独自プレイヤーでは、シミュレーター側の追従遅れを検出できなかった)。

このクラスはrclpy.node.Nodeを継承せず、外部(py_trees_ros.trees.BehaviourTree.setup()が
内部生成する、またはmain()で明示的に渡すノード)から`node`を受け取る。behaviour側が
tickのたびに再購読・再生成しなくて済むよう、TF listener・publisher・subscription・
action clientはこのクラスの初期化時に1度だけ作る。
"""

import math
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from common.target_object_shape import OBJECT_SIZE_M as TARGET_SIZE
from common.target_object_shape import TABLE_HEIGHT_M
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit.planning import MoveItPy
from moveit.utils import create_params_file_from_dict
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
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
GRIPPER_OPEN_POSITIONS = [0.035, 0.035]
# target_objectの半径(TARGET_SIZE/2=0.025m)より深く閉じる指令にすると、位置制御の
# フィンガーが軽量なキューブを弾き飛ばして指の間から押し出し、何も挟まないまま
# 0まで閉じきってしまう(/joint_statesの実測値で接触・押し出しの挙動を確認済み)。
# 半径よりわずかに手前で止めることで、キューブを押し出さずに挟み込む力を残す。
GRIPPER_CLOSED_POSITIONS = [0.01, 0.01]

APPROACH_HEIGHT = 0.10
PLACE_POSITION = (0.3, 0.3, TABLE_HEIGHT_M + TARGET_SIZE / 2.0)

# Ref: edge/workspace/.pixi/envs/default/.../isaacsim/exts/isaacsim.robot.experimental.manipulators.examples/
#      isaacsim/robot/experimental/manipulators/examples/franka/franka.py Franka.get_downward_orientation()
# 同メソッドが返す(w,x,y,z)=[0,1,0,0]をgeometry_msgs/Quaternion(x,y,z,w)の並びに変換したもの。
# panda_link8のローカルZ軸をworldの-Z(真下)へ向ける姿勢。
DOWNWARD_ORIENTATION = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)

# panda_link8(フランジ)からフィンガー先端(把持点)までのオフセット。Frankaハンドの
# 標準的な長さ(約0.1034m)による近似値。
FLANGE_TO_FINGERTIP_Z = 0.1034

# move_to(verify_and_correct=True)で許容する、実行後のフランジ位置と指令位置のズレ。
CORRECTION_TOLERANCE_M = 0.01
MAX_CORRECTION_ATTEMPTS = 2

# move_to(yaw_free=True)で使う、真下向き(ロール・ピッチに相当するX/Y軸)の姿勢許容誤差。
YAW_FREE_TILT_TOLERANCE_RAD = 0.05
# 同、鉛直軸(yaw)まわりの許容誤差。grasp_orientation_for_yaw()が指令するyawちょうどでは、
# 水平リーチ0.5m付近でjoint2/6がトルク上限に張り付き指令姿勢に届かないことがある。
# yawを緩めてプランナーに選択の余地を残すが、TRAC-IK/OMPLはトルクを考慮せず解を選ぶため、
# これだけでは解消しきらない(TABLE_HEIGHT_M側の高さ調整と組み合わせても未解消)。
YAW_FREE_YAW_TOLERANCE_RAD = math.pi


def downward_orientation_for_yaw(yaw: float) -> Quaternion:
    """DOWNWARD_ORIENTATION(panda_link8のローカルZ+をworldの-Zへ向ける姿勢)を、world Z軸周りにyaw回転させる。"""
    return Quaternion(x=math.cos(yaw / 2.0), y=math.sin(yaw / 2.0), z=0.0, w=0.0)


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
    return downward_orientation_for_yaw(yaw_mod)


class RobotInterface:
    def __init__(self, node: Node) -> None:
        self.node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._ground_truth_target_pose: Pose | None = None
        node.create_subscription(TFMessage, GROUND_TRUTH_TF_TOPIC, self._on_ground_truth_tf, 10)
        # gripper_moveit_controllers.yamlで設定するpanda_hand_controller(GripperActionController)
        # のアクション名(vp-controller-manager起動が別途必要)。
        self._gripper_action_client = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")

        # vp-controller-manager(controller_manager+spawner)はvp-run-yolo等のMakefileターゲットで
        # 他のノードと並行してバックグラウンド起動されるため、コントローラのアクティブ化が
        # このノードの起動より遅れることがある(既知のros2_control起動レース条件)。
        # moveit_pyは内部でpanda_arm_controllerのFollowJointTrajectoryアクションクライアントを
        # 遅延生成するため、事前に素のActionClientでサーバーの起動を待ってから進める。
        arm_trajectory_action_client = ActionClient(
            node, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory"
        )
        if not arm_trajectory_action_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(
                "panda_arm_controllerのFollowJointTrajectoryアクションサーバーが見つかりません"
                "(vp-controller-managerが起動しているか確認してください)"
            )
        arm_trajectory_action_client.destroy()

        # Ref: https://github.com/moveit/moveit2/issues/2409#issuecomment-1753090620
        # moveit_resources_panda_moveit_configにはmoveit_cpp.yamlが同梱されておらず、
        # .moveit_cpp(file_path=...)を明示的に呼ばない限りMoveItPyがplanning_pipelinesを
        # 解決できずに初期化失敗する(Failed to load any planning pipelines)ため、
        # 本パッケージ側で用意したmoveit_cpp.yamlを明示的に読み込む。
        moveit_cpp_yaml_path = get_package_share_directory("vision_picking") + "/config/moveit_cpp.yaml"
        # moveit_resources_panda_moveit_config同梱のpanda.urdf.xacroはhand側もros2_controlに
        # 含めてしまい、mimic関節panda_finger_joint2の扱いでtopic_based_ros2_controlが
        # メモリを壊してcontroller_manager全体を不安定化させる。本パッケージ側の
        # panda_isaac.urdf.xacro(arm側のros2_controlのみ)を絶対パスで指定する。
        # グリッパーはgripper_to_isaac_node経由(vp-controller-manager launch側と
        # 同じmappings、そちらのros2_control_hardware_typeと必ず揃える)。
        # trajectory_executionは本パッケージ側で用意したgripper_moveit_controllers.yaml
        # (moveit_resources_panda_moveit_config同梱版のコピーがベース)を明示的に指定する。
        # アーム(FollowJointTrajectory)だけでなくグリッパー(GripperCommand)分の
        # コントローラ定義も読み込む(既定のmoveit_controllers.yamlはアーム分のみ)のに加え、
        # allowed_start_toleranceも緩めている。
        trajectory_execution_yaml_path = (
            get_package_share_directory("vision_picking") + "/config/gripper_moveit_controllers.yaml"
        )
        # MoveItConfigsBuilderのfile_path引数は自身のpackage_name
        # (moveit_resources_panda_moveit_config)基準で解決されるため、相対パスのままだと
        # 同梱版のkinematics.yaml(KDL)が常に使われてしまう。本パッケージ側の上書き版
        # (TRAC-IK)を使うには絶対パスで渡す必要がある。
        kinematics_yaml_path = get_package_share_directory("vision_picking") + "/config/kinematics.yaml"
        panda_isaac_urdf_xacro_path = get_package_share_directory("vision_picking") + "/config/panda_isaac.urdf.xacro"
        moveit_config = (
            MoveItConfigsBuilder("panda", package_name="moveit_resources_panda_moveit_config")
            .robot_description(
                file_path=panda_isaac_urdf_xacro_path,
                mappings={
                    "ros2_control_hardware_type": "isaac",
                    "initial_positions_file": get_package_share_directory("moveit_resources_panda_moveit_config")
                    + "/config/initial_positions.yaml",
                },
            )
            .robot_description_kinematics(file_path=kinematics_yaml_path)
            .trajectory_execution(file_path=trajectory_execution_yaml_path)
            .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
            .planning_scene_monitor(
                publish_robot_description=True,
                publish_robot_description_semantic=True,
            )
            .moveit_cpp(file_path=moveit_cpp_yaml_path)
            .to_moveit_configs()
        )
        # Ref: https://github.com/moveit/moveit2/issues/2940#issuecomment-2401302214
        # シミュレーター側は/ground_truth/tf・/clock等をシミュレーション時刻でタイムスタンプして
        # おり、MoveItPyもuse_sim_timeを設定しないとこれらと時刻の整合が取れない。
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
        # MoveItPy自身もmoveit_simple_controller_manager経由でpanda_arm_controller/
        # panda_hand_controller向けの独自のアクションクライアントをこの時点で生成する。
        # 上のarm_trajectory_action_clientでのサーバー存在確認はこのクライアントとは別物なので、
        # サーバー側が既に存在していても、MoveItPy側のクライアントがDDSのdiscoveryを
        # 完了しきる前にexecute()を呼ぶと「Action client not connected」で即座に失敗する
        # ことがある。生成直後に少し待ち、discoveryが追いつく余地を作る。
        time.sleep(3.0)
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
        self,
        position_xyz: tuple[float, float, float],
        orientation: Quaternion = DOWNWARD_ORIENTATION,
        verify_and_correct: bool = False,
        yaw_free: bool = False,
    ) -> None:
        # 回転→水平移動→下降と複数回move_to()を連続させる経路では、各回はplan()自体が
        # ほぼ正確でも、実機的な物理応答の遅れによる数mm〜cm単位の実行誤差が1回ごとに
        # 乗り、連続することで累積する(閉ループの手先姿勢フィードバックで補正するのは
        # 一般的な手法、Ref: https://arxiv.org/pdf/2502.07472)。掴む直前の下降のように
        # わずかなズレが結果を左右するステップでは、実行後に実際の到達位置を確認し、
        # ズレていれば同じ目標へもう一度move_to()して閉じる。
        self._move_to_once(position_xyz, orientation, yaw_free=yaw_free)
        if not verify_and_correct:
            return
        for _ in range(MAX_CORRECTION_ATTEMPTS):
            error_m = self._flange_position_error_m(position_xyz)
            if error_m <= CORRECTION_TOLERANCE_M:
                return
            try:
                self._move_to_once(position_xyz, orientation, yaw_free=yaw_free)
            except RuntimeError as e:
                # 補正の再試行自体が失敗しても、直前の(不完全だが到達済みの)姿勢のまま
                # 後続処理(把持)へ進める方が、ここで全体を失敗させるより良い結果になる。
                self.node.get_logger().warning(f"補正move_to()に失敗、直前の姿勢のまま続行します: {e}")
                return

    def _yaw_free_goal_constraints(
        self, position_xyz: tuple[float, float, float], orientation: Quaternion
    ) -> Constraints:
        constraints = Constraints()

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = WORLD_FRAME
        position_constraint.link_name = END_EFFECTOR_LINK
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [CORRECTION_TOLERANCE_M]
        region_pose = Pose()
        region_pose.position.x = position_xyz[0]
        region_pose.position.y = position_xyz[1]
        region_pose.position.z = position_xyz[2] + FLANGE_TO_FINGERTIP_Z
        region_pose.orientation.w = 1.0
        position_constraint.constraint_region.primitives.append(region)
        position_constraint.constraint_region.primitive_poses.append(region_pose)
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = WORLD_FRAME
        orientation_constraint.link_name = END_EFFECTOR_LINK
        orientation_constraint.orientation = orientation
        orientation_constraint.absolute_x_axis_tolerance = YAW_FREE_TILT_TOLERANCE_RAD
        orientation_constraint.absolute_y_axis_tolerance = YAW_FREE_TILT_TOLERANCE_RAD
        orientation_constraint.absolute_z_axis_tolerance = YAW_FREE_YAW_TOLERANCE_RAD
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

        return constraints

    def _move_to_once(
        self, position_xyz: tuple[float, float, float], orientation: Quaternion, yaw_free: bool = False
    ) -> None:
        # 直前のmove_to()の完了直後は、/joint_states(joint_state_broadcaster配信)が
        # まだ最終姿勢を反映していないことがあり、start_state_to_current_state()が古い状態を
        # 拾うと、後段のtrajectory_execution_manager側の検証(allowed_start_tolerance)で
        # 「計画の開始点が現在状態とズレている」と判定され実行そのものが拒否される。
        # 直近以降に届いた新しい状態を待ってから計画する。
        self._psm.wait_for_current_robot_state(self.node.get_clock().now(), 1.0)
        self._arm.set_start_state_to_current_state()
        if yaw_free:
            self._arm.set_goal_state(
                motion_plan_constraints=[self._yaw_free_goal_constraints(position_xyz, orientation)]
            )
        else:
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = WORLD_FRAME
            pose_goal.pose.position.x = position_xyz[0]
            pose_goal.pose.position.y = position_xyz[1]
            pose_goal.pose.position.z = position_xyz[2] + FLANGE_TO_FINGERTIP_Z
            pose_goal.pose.orientation = orientation
            self._arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=END_EFFECTOR_LINK)
        plan_result = self._arm.plan()
        if not plan_result:
            raise RuntimeError(f"軌道計画に失敗しました: target={position_xyz}")
        # execute()はcontroller_manager側のFollowJointTrajectoryアクションが完了を
        # 報告するまでブロックする。計画時刻通りに関節コマンドをpublishするだけの
        # 独自プレイヤーと違い、シミュレーター側が実際に計画通りの姿勢へ追従できたかを
        # 実行結果で確認できる。
        if not self.moveit_py.execute(plan_result.trajectory, controllers=[]):
            raise RuntimeError(f"軌道の実行に失敗しました: target={position_xyz}")

    def _flange_position_error_m(self, position_xyz: tuple[float, float, float]) -> float:
        self._psm.wait_for_current_robot_state(self.node.get_clock().now(), 1.0)
        with self._psm.read_only() as scene:
            actual = scene.current_state.get_pose(END_EFFECTOR_LINK).position
        target_z = position_xyz[2] + FLANGE_TO_FINGERTIP_Z
        return math.sqrt(
            (actual.x - position_xyz[0]) ** 2 + (actual.y - position_xyz[1]) ** 2 + (actual.z - target_z) ** 2
        )

    def _wait_for_future(self, future, timeout_sec: float):
        # 呼び出し元(picking_controller_node.py/stacking_controller_node.py)は既に
        # 別スレッドでnodeをspinしているため、ここでさらにspinすると同じexecutorを
        # 二重にspinすることになり危険(コールバックの取りこぼし・デッドロックの原因になる)。
        # 別スレッド側のspinがfutureを完了させるのを、spinせずポーリングだけで待つ。
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError("アクションの完了待ちがタイムアウトしました")
            time.sleep(0.05)
        return future.result()

    def set_gripper(self, positions: list[float]) -> None:
        if not self._gripper_action_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("panda_hand_controllerのGripperCommandアクションサーバーが見つかりません")
        goal = GripperCommand.Goal()
        goal.command.position = positions[0]
        goal.command.max_effort = 0.0
        goal_handle = self._wait_for_future(self._gripper_action_client.send_goal_async(goal), timeout_sec=5.0)
        if not goal_handle.accepted:
            raise RuntimeError("GripperCommandのゴールがpanda_hand_controllerに拒否されました")
        # stall_timeout(0.5秒)は物理的にstallしたと判定されてからの猶予でしかなく、
        # そこに至るまでの実行完了フィードバック自体もIsaac Sim側の更新頻度に依存するため、
        # 5秒では不足することがある。
        self._wait_for_future(goal_handle.get_result_async(), timeout_sec=10.0)

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
