#!/usr/bin/env python3
"""Isaac Simを起動し、Panda + target_objectをロードしてROS2 Bridgeを自動配線する。

SimulationAppはisaacsim配下の他モジュールをimportする前に必ず生成する必要があるため、
このスクリプトではargparse以外の一切のisaacsim/omni importをSimulationApp生成後に置く。
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="GUIを表示せずに起動する")
    return parser.parse_args()


def build_action_graph(panda_prim_path: str, target_object_prim_path: str):
    """ROS2Context / JointState Pub-Sub / ArticulationController / 物体GT位置姿勢配線を構築する。

    target_objectの真値TFは最終的な`/tf`ではなく`/ground_truth/tf`にpublishする。
    `vision_picking`パッケージの`gt_tf_publisher_node`がこれをsubscribeして
    world -> target_objectとして`/tf`に再publishする構成にすることで、
    認識方式を変える場合でも`/tf`へのpublisher側を差し替えるだけで済むようにするため。
    """
    import omni.graph.core as og

    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ROS2PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("ROS2PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("ROS2SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ROS2PublishGtTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2PublishGtTransformTree.inputs:execIn"),
                ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishJointState.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2SubscribeJointState.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishGtTransformTree.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "ROS2PublishGtTransformTree.inputs:timeStamp"),
                ("ROS2SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("ROS2SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("ROS2SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("ROS2SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ROS2PublishJointState.inputs:targetPrim", panda_prim_path),
                ("ArticulationController.inputs:targetPrim", panda_prim_path),
                ("ROS2PublishJointState.inputs:topicName", "/joint_states"),
                ("ROS2SubscribeJointState.inputs:topicName", "/joint_command"),
                ("ROS2PublishGtTransformTree.inputs:targetPrims", [target_object_prim_path]),
                ("ROS2PublishGtTransformTree.inputs:topicName", "/ground_truth/tf"),
            ],
        },
    )


def main():
    args = parse_args()

    import os

    from isaacsim import SimulationApp

    # isaacsim.exp.full.kitを明示することでGUI一式を有効にする。
    experience = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.full.kit")
    simulation_app = SimulationApp({"headless": args.headless}, experience=experience)

    # SimulationApp生成後でないとimportできない。
    import omni.kit.app

    # 以下の拡張機能はisaacsim.exp.base.python.kitのデフォルト有効セットに含まれないため、
    # importやOmniGraphノード生成が失敗しないよう明示的に有効化する。
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_id in (
        "isaacsim.robot.experimental.manipulators.examples",
        "isaacsim.ros2.bridge",
        "isaacsim.core.nodes",
    ):
        extension_manager.set_extension_enabled_immediate(ext_id, True)

    import isaacsim.core.experimental.utils.stage as stage_utils
    import numpy as np
    import omni.timeline
    from isaacsim.core.experimental.objects import Cube
    from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
    from isaacsim.robot.experimental.manipulators.examples.franka import Franka
    from isaacsim.storage.native import get_assets_root_path

    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/ground",
    )

    Franka(robot_path="/World/panda")

    target_shape = Cube(
        paths="/World/target_object",
        positions=np.array([[0.5, 0.0, 0.05]]),
        sizes=[0.05],
        reset_xform_op_properties=True,
    )
    GeomPrim(paths=target_shape.paths, apply_collision_apis=True)
    RigidPrim(paths=target_shape.paths)

    build_action_graph(panda_prim_path="/World/panda", target_object_prim_path="/World/target_object")

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
