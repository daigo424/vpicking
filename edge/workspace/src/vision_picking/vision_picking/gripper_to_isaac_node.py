#!/usr/bin/env python3
"""グリッパーの/isaac_joint_commands直接ブリッジ。

panda_hand.ros2_control.xacro(topic_based_ros2_control経由)は、command_interfaceを持たない
mimic関節panda_finger_joint2を扱う際にjoint_state.positionの範囲外へ書き込み、同じ
ros2_control_nodeプロセス内の他のros2_control(panda_arm_controller含む)のメモリを壊して
不安定化させる(Ref: https://github.com/isaac-sim/IsaacSim-ros_workspaces
isaac_moveit/config/panda_isaac.urdf.xacroのコメント)。そのためグリッパーはros2_control
経由にせず、このノードが/isaac_joint_commandsへ直接コマンドを送り、GripperCommandアクション
サーバーを自前で提供する。picking_robot_interface.RobotInterfaceが繋ぐアクション名
(/panda_hand_controller/gripper_cmd)はgripper_moveit_controllers.yaml上の定義と揃えている。
"""

import threading
import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
GRIPPER_COMMAND_TOPIC = "/panda_hand_controller/gripper_cmd"
ISAAC_JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"
ISAAC_JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_STATES_TOPIC = "/joint_states"
COMMAND_PUBLISH_PERIOD_SEC = 0.05
SETTLE_WAIT_SEC = 1.0


class GripperToIsaacNode(Node):
    def __init__(self) -> None:
        super().__init__("gripper_to_isaac_node")
        self._target_position = 0.04
        self._target_lock = threading.Lock()

        self._isaac_command_pub = self.create_publisher(JointState, ISAAC_JOINT_COMMANDS_TOPIC, 10)
        # panda_hand側がros2_controlに載っていないため、joint_state_broadcasterは指の状態を
        # 配信しない。MoveItのplanning scene monitorが把持状態を認識できるよう、
        # /isaac_joint_statesから指の位置を/joint_statesへミラーする。
        self._joint_states_pub = self.create_publisher(JointState, JOINT_STATES_TOPIC, 10)
        self.create_subscription(JointState, ISAAC_JOINT_STATES_TOPIC, self._on_isaac_joint_states, 10)
        self.create_timer(COMMAND_PUBLISH_PERIOD_SEC, self._publish_isaac_command)

        self._action_server = ActionServer(
            self,
            GripperCommand,
            GRIPPER_COMMAND_TOPIC,
            self._execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )

    def _publish_isaac_command(self) -> None:
        with self._target_lock:
            position = self._target_position
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(FINGER_JOINT_NAMES)
        msg.position = [position, position]
        self._isaac_command_pub.publish(msg)

    def _on_isaac_joint_states(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        finger_positions = [positions.get(name) for name in FINGER_JOINT_NAMES]
        if any(p is None for p in finger_positions):
            return
        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = list(FINGER_JOINT_NAMES)
        out.position = finger_positions
        self._joint_states_pub.publish(out)

    async def _execute_callback(self, goal_handle: ServerGoalHandle):
        with self._target_lock:
            self._target_position = goal_handle.request.command.position

        # topic_based_ros2_control(GripperActionController)のstall検出を代替するものは無く、
        # 実際に掴めたかはpicking_robot_interface.verify_grasped()がground truthで別途検証する。
        # ここでは指令を送って物理的に動き切るのを待つだけに徹する。
        time.sleep(SETTLE_WAIT_SEC)

        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = goal_handle.request.command.position
        result.reached_goal = True
        result.stalled = False
        return result


def main() -> None:
    rclpy.init()
    node = GripperToIsaacNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
