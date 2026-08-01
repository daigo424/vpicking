#!/usr/bin/env python3
"""world -> target_objectのTFを取得し、moveit_pyでIK・軌道計画してアプローチ→下降→
把持→上昇→プレイスの一連動作を実行するピッキングコントローラ。

moveit_py標準のtrajectory_execution(ros2_controlのFollowJointTrajectoryアクション)は
Isaac Sim側に存在せず、run_simulation.pyのOmniGraphは/joint_command(sensor_msgs/JointState)を
直接subscribeするだけの構成になっている。そのため、計画したtrajectoryのwaypointを
このノード自身がタイマー無しの逐次sleepループで/joint_commandへpublishする、
簡易的なtrajectoryプレイヤーを実装している。
"""

import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit.planning import MoveItPy
from moveit.utils import create_params_file_from_dict
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

WORLD_FRAME = "world"
TARGET_FRAME = "target_object"
ARM_GROUP = "panda_arm"
END_EFFECTOR_LINK = "panda_link8"
JOINT_COMMAND_TOPIC = "/joint_command"
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_OPEN_POSITIONS = [0.035, 0.035]
GRIPPER_CLOSED_POSITIONS = [0.0, 0.0]

TARGET_SIZE = 0.05
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


class PickingControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("picking_controller_node")
        self._joint_command_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

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

    def _lookup_target_pose(self) -> Pose:
        while rclpy.ok():
            try:
                transform = self._tf_buffer.lookup_transform(WORLD_FRAME, TARGET_FRAME, rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().info(f"{TARGET_FRAME}のTFを待機中...")
                time.sleep(0.5)
                continue
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = DOWNWARD_ORIENTATION
            return pose
        raise RuntimeError("rclpy shutdown中にTF取得が中断されました")

    def _register_collision_object(self, frame_id: str, object_id: str, pose: Pose) -> None:
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

    def _remove_collision_object(self, object_id: str) -> None:
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = WORLD_FRAME
        obj.operation = CollisionObject.REMOVE
        self._psm.process_collision_object(collision_object_msg=obj)

    def _move_to(self, position_xyz: tuple[float, float, float]) -> None:
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = WORLD_FRAME
        pose_goal.pose.position.x = position_xyz[0]
        pose_goal.pose.position.y = position_xyz[1]
        pose_goal.pose.position.z = position_xyz[2] + FLANGE_TO_FINGERTIP_Z
        pose_goal.pose.orientation = DOWNWARD_ORIENTATION

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
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = list(point.positions)
            self._joint_command_pub.publish(msg)

    def _set_gripper(self, positions: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = GRIPPER_JOINT_NAMES
        msg.position = positions
        self._joint_command_pub.publish(msg)
        time.sleep(2.0)

    def _attach_target(self) -> None:
        # 下降前にプランニングシーンからtarget_objectを外している(_remove_collision_object)ため、
        # ジオメトリを持たないAttachedCollisionObjectをADDしても中身が空になってしまう。
        # end-effectorからのローカル姿勢込みでジオメトリを再定義してアタッチする。
        attached = AttachedCollisionObject()
        attached.link_name = END_EFFECTOR_LINK
        attached.object.id = TARGET_FRAME
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

    def _detach_target(self) -> None:
        attached = AttachedCollisionObject()
        attached.link_name = END_EFFECTOR_LINK
        attached.object.id = TARGET_FRAME
        attached.object.operation = CollisionObject.REMOVE
        self._psm.process_attached_collision_object(attached_collision_object_msg=attached)

    def run_pick_and_place(self) -> None:
        target_pose = self._lookup_target_pose()
        self._register_collision_object(WORLD_FRAME, TARGET_FRAME, target_pose)

        pick_x, pick_y, pick_z = target_pose.position.x, target_pose.position.y, target_pose.position.z
        place_x, place_y, place_z = PLACE_POSITION

        self.get_logger().info("アプローチ")
        self._move_to((pick_x, pick_y, pick_z + APPROACH_HEIGHT))
        self._set_gripper(GRIPPER_OPEN_POSITIONS)

        self.get_logger().info("下降")
        # target_objectをプランニングシーンに登録したまま指先をその中心へ合わせるIKを解こうとすると、
        # target_object自体との衝突により有効な目標姿勢が見つからずGOAL_STATE_INVALIDになるため、
        # 掴みに行く直前だけ一時的に外す(把持後はattached collision objectとして登録し直す)。
        self._remove_collision_object(TARGET_FRAME)
        self._move_to((pick_x, pick_y, pick_z))

        self.get_logger().info("把持")
        self._set_gripper(GRIPPER_CLOSED_POSITIONS)
        self._attach_target()

        self.get_logger().info("上昇")
        self._move_to((pick_x, pick_y, pick_z + APPROACH_HEIGHT))

        self.get_logger().info("プレイス位置上空へ移動")
        self._move_to((place_x, place_y, place_z + APPROACH_HEIGHT))

        self.get_logger().info("プレイス降下")
        self._move_to((place_x, place_y, place_z))

        self.get_logger().info("リリース")
        self._set_gripper(GRIPPER_OPEN_POSITIONS)
        # Ref: https://github.com/moveit/moveit2/issues/1070
        # AttachedCollisionObjectのREMOVE操作は実際には削除ではなく、アタッチを解除して
        # 「その時点のグリッパー位置のまま」ワールドの衝突オブジェクトへ自動変換するというMoveIt2の
        # 既知の挙動。指がまだ物体のすぐ脇にある位置でこれが起きるため、そのまま次のplan()に進むと
        # 指の開き幅と物体サイズのマージン次第でCheckStartStateCollisionに失敗する。
        # 一旦明示的に削除し、退避で十分離れた後にあらためて正しい位置へ登録し直す。
        self._detach_target()
        self._remove_collision_object(TARGET_FRAME)

        self.get_logger().info("退避")
        # target_objectをフリーの衝突オブジェクトとして登録するのは、指がまだ物体のすぐ脇にある
        # このタイミングではなく、退避で十分離れた後にする。指を開いた直後の位置にオブジェクトを
        # 再登録すると、指の開き幅(0.07m)と物体サイズ(0.05m)の余裕がほぼ無く、指リンクの
        # 形状マージン次第で接触判定になりCheckStartStateCollisionが失敗するため。
        self._move_to((place_x, place_y, place_z + APPROACH_HEIGHT))
        place_pose = Pose()
        place_pose.position.x = place_x
        place_pose.position.y = place_y
        place_pose.position.z = place_z
        place_pose.orientation = target_pose.orientation
        self._register_collision_object(WORLD_FRAME, TARGET_FRAME, place_pose)

        self.get_logger().info("ピック&プレイス完了")


def main() -> None:
    rclpy.init()
    node = PickingControllerNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run_pick_and_place()
    finally:
        # spin_threadがexecutorをスピンしたままmoveit_py.shutdown()/destroy_node()を呼ぶと、
        # 実行中のコールバックとリソース解放が競合してabortする。rclpy.shutdown()を先に呼んで
        # spin()側にExternalShutdownExceptionを送出させ、スレッド終了(join)を待ってから
        # 後片付けする。
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
        node.moveit_py.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
