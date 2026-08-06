#!/usr/bin/env python3
"""ros2_control(controller_manager)一式を起動する。

panda_isaac.urdf.xacro(vision_picking同梱)はarm側(7関節)のみをros2_controlに載せる。
moveit_resources_panda_moveit_config同梱のpanda.urdf.xacroはhand側(グリッパー)も
ros2_controlに含めるが、mimic関節panda_finger_joint2の扱いでtopic_based_ros2_controlが
メモリを壊しcontroller_manager全体を不安定化させるため使わない(panda_isaac.urdf.xacro
のコメント参照)。グリッパーはgripper_to_isaac_nodeが/isaac_joint_commandsへ直接コマンドを
送り、GripperCommandアクションサーバーも自前で提供する。

標準の/joint_states・FollowJointTrajectoryアクションはjoint_state_broadcaster・
joint_trajectory_controllerがこのros2_control層の上に提供するため、picking_robot_interface.py
側はmoveit_py標準のexecute()をそのまま使える。
"""

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    vision_picking_share = FindPackageShare("vision_picking")
    panda_moveit_config_share = FindPackageShare("moveit_resources_panda_moveit_config")
    panda_urdf_xacro = PathJoinSubstitution([vision_picking_share, "config", "panda_isaac.urdf.xacro"])
    initial_positions_file = PathJoinSubstitution([panda_moveit_config_share, "config", "initial_positions.yaml"])
    ros2_controllers_yaml = PathJoinSubstitution([panda_moveit_config_share, "config", "ros2_controllers.yaml"])

    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    panda_urdf_xacro,
                    " ros2_control_hardware_type:=isaac",
                    " initial_positions_file:=",
                    initial_positions_file,
                ]
            ),
            value_type=str,
        )
    }

    # ros2_control_nodeはrobot_descriptionをパラメータではなく/robot_descriptionトピック
    # (transient local QoS)から受け取る仕様のため、robot_state_publisherを併せて起動する。
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": True}],
        output="screen",
    )

    # panda.srdfのvirtual_joint(world -> panda_link0)はfloating型で、これを更新する
    # localization等が存在しないため、robot_state_publisherの/tfにworldフレームへの
    # エッジが実際には存在しない。Isaac Sim側でpanda_link0がworld原点にあることに
    # 合わせ、恒等変換を明示的にpublishしてTFツリーを繋げる。
    world_to_panda_link0_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
        parameters=[{"use_sim_time": True}],
    )

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        # use_sim_timeをここで設定すると、controller_managerの制御ループが/clockトリガー駆動に
        # 切り替わり、コントローラのactivate確認(switch_controllerサービス)が
        # 30秒のswitch-timeoutを与えても完了しない(ros2_control_node側の既知の相性問題)。
        # 実時間で動かし、Isaac Sim側との時刻整合はjoint_state_broadcasterが再配信する
        # /joint_statesのタイムスタンプ側の問題として別途扱う。
        parameters=[robot_description, ros2_controllers_yaml],
        output="screen",
    )

    # panda_arm_controller(joint_trajectory_controller)はallow_nonzero_velocity_at_trajectory_end
    # が既定trueのため、速度が残ったまま(=物理的に到達しきる前に)到達成功と報告されることがある
    # (panda_arm_controller_overrides.yaml参照)。
    panda_arm_controller_overrides = PathJoinSubstitution(
        [FindPackageShare("vision_picking"), "config", "panda_arm_controller_overrides.yaml"]
    )
    spawners = [
        Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"]),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["panda_arm_controller", "--param-file", panda_arm_controller_overrides],
        ),
    ]

    # use_sim_timeを指定しないと、このノードが/joint_statesへ付与するタイムスタンプが
    # 実時間(壁時計)基準になり、他ノードのシム時刻基準のタイムスタンプと食い違って
    # TF_OLD_DATA警告が大量発生する(robot_state_publisherもuse_sim_time=Trueのため、
    # シム時刻より新しい実時間タイムスタンプを「未来」と誤認しうる)。
    gripper_to_isaac_node = Node(
        package="vision_picking", executable="gripper_to_isaac_node", parameters=[{"use_sim_time": True}]
    )

    return LaunchDescription(
        [robot_state_publisher_node, world_to_panda_link0_tf_node, controller_manager_node, *spawners, gripper_to_isaac_node]
    )
