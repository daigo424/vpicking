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


def build_action_graph(panda_prim_path: str, target_object_prim_path: str, camera_prim_path: str):
    """ROS2Context / JointState Pub-Sub / ArticulationController / 物体GT位置姿勢配線 / カメラ配線を構築する。

    target_objectとcameraの真値TFは最終的な`/tf`ではなく`/ground_truth/tf`にpublishする。
    `vision_picking`パッケージの`gt_tf_publisher_node`がこれをsubscribeして
    world -> target_objectとして`/tf`に再publishする構成にすることで、
    認識方式を変える場合でも`/tf`へのpublisher側を差し替えるだけで済むようにするため。
    カメラ画像も同様に、OmniGraph側の実装依存のトピック名(/sim_camera/*)で一旦publishし、
    認識ノード側が依存する安定したトピック名への変換は`vision_picking`パッケージ側の
    ノードに委ねる。
    """
    import omni.graph.core as og

    # Ref: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/migration_guides/isaac_sim_6_0/ros2_omnigraph_migration.html
    # Isaac Sim 6.0でROS2PublishTransformTree/ROS2PublishJointStateにtargetPrim(s)を直接渡す方式が
    # 非推奨になり、IsaacComputeTransformTree/IsaacReadJointStateが計算した値を明示的に渡す構成が
    # 推奨されているため、その配線に従っている。
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ROS2PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
                ("ROS2PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("ROS2SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ComputeGtTransformTree", "isaacsim.core.nodes.IsaacComputeTransformTree"),
                ("ROS2PublishGtTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("CreateCameraRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("ROS2PublishCameraRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2PublishCameraDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ComputeGtTransformTree.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "CreateCameraRenderProduct.inputs:execIn"),
                ("ReadJointState.outputs:execOut", "ROS2PublishJointState.inputs:execIn"),
                ("ReadJointState.outputs:jointNames", "ROS2PublishJointState.inputs:jointNames"),
                ("ReadJointState.outputs:jointPositions", "ROS2PublishJointState.inputs:jointPositions"),
                ("ReadJointState.outputs:jointVelocities", "ROS2PublishJointState.inputs:jointVelocities"),
                ("ReadJointState.outputs:jointEfforts", "ROS2PublishJointState.inputs:jointEfforts"),
                ("ReadJointState.outputs:jointDofTypes", "ROS2PublishJointState.inputs:jointDofTypes"),
                ("ReadJointState.outputs:sensorTime", "ROS2PublishJointState.inputs:sensorTime"),
                ("ReadJointState.outputs:stageMetersPerUnit", "ROS2PublishJointState.inputs:stageMetersPerUnit"),
                ("ComputeGtTransformTree.outputs:execOut", "ROS2PublishGtTransformTree.inputs:execIn"),
                ("ComputeGtTransformTree.outputs:parentFrames", "ROS2PublishGtTransformTree.inputs:parentFrames"),
                ("ComputeGtTransformTree.outputs:childFrames", "ROS2PublishGtTransformTree.inputs:childFrames"),
                ("ComputeGtTransformTree.outputs:translations", "ROS2PublishGtTransformTree.inputs:translations"),
                ("ComputeGtTransformTree.outputs:orientations", "ROS2PublishGtTransformTree.inputs:orientations"),
                ("CreateCameraRenderProduct.outputs:execOut", "ROS2PublishCameraRgb.inputs:execIn"),
                ("CreateCameraRenderProduct.outputs:execOut", "ROS2PublishCameraDepth.inputs:execIn"),
                ("CreateCameraRenderProduct.outputs:execOut", "ROS2PublishCameraInfo.inputs:execIn"),
                ("CreateCameraRenderProduct.outputs:renderProductPath", "ROS2PublishCameraRgb.inputs:renderProductPath"),
                ("CreateCameraRenderProduct.outputs:renderProductPath", "ROS2PublishCameraDepth.inputs:renderProductPath"),
                ("CreateCameraRenderProduct.outputs:renderProductPath", "ROS2PublishCameraInfo.inputs:renderProductPath"),
                ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishJointState.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2SubscribeJointState.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishGtTransformTree.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishCameraRgb.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishCameraDepth.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishCameraInfo.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "ROS2PublishGtTransformTree.inputs:timeStamp"),
                ("ROS2SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("ROS2SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("ROS2SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("ROS2SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ReadJointState.inputs:prim", panda_prim_path),
                ("ArticulationController.inputs:targetPrim", panda_prim_path),
                ("ROS2PublishJointState.inputs:topicName", "/joint_states"),
                ("ROS2SubscribeJointState.inputs:topicName", "/joint_command"),
                ("ComputeGtTransformTree.inputs:targetPrims", [target_object_prim_path, camera_prim_path]),
                ("ROS2PublishGtTransformTree.inputs:topicName", "/ground_truth/tf"),
                ("CreateCameraRenderProduct.inputs:cameraPrim", camera_prim_path),
                ("CreateCameraRenderProduct.inputs:width", 640),
                ("CreateCameraRenderProduct.inputs:height", 480),
                ("ROS2PublishCameraRgb.inputs:type", "rgb"),
                ("ROS2PublishCameraRgb.inputs:topicName", "/sim_camera/rgb"),
                ("ROS2PublishCameraRgb.inputs:frameId", "camera"),
                ("ROS2PublishCameraDepth.inputs:type", "depth"),
                ("ROS2PublishCameraDepth.inputs:topicName", "/sim_camera/depth"),
                ("ROS2PublishCameraDepth.inputs:frameId", "camera"),
                ("ROS2PublishCameraInfo.inputs:topicName", "/sim_camera/camera_info"),
                ("ROS2PublishCameraInfo.inputs:frameId", "camera"),
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
    from isaacsim.core.experimental.materials import RigidBodyMaterial
    from isaacsim.core.experimental.objects import Camera, Cube
    from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
    from isaacsim.robot.experimental.manipulators.examples.franka import Franka
    from isaacsim.storage.native import get_assets_root_path

    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/ground",
    )

    Franka(robot_path="/World/panda")

    # TARGET_OBJECT_X/Y/YAW環境変数が指定されていればそれを使う。学習データ収集用に
    # 物体の初期位置・向きを実行のたびに変えられるようにするためのフックで、未指定時は
    # 従来通り固定位置・向きになる。
    target_x = float(os.environ.get("TARGET_OBJECT_X", 0.5))
    target_y = float(os.environ.get("TARGET_OBJECT_Y", 0.0))
    target_yaw = float(os.environ.get("TARGET_OBJECT_YAW", 0.0))
    target_orientation = [np.cos(target_yaw / 2.0), 0.0, 0.0, np.sin(target_yaw / 2.0)]
    target_shape = Cube(
        paths="/World/target_object",
        positions=np.array([[target_x, target_y, 0.05]]),
        orientations=np.array([target_orientation]),
        sizes=[0.05],
        reset_xform_op_properties=True,
    )
    target_geom = GeomPrim(paths=target_shape.paths, apply_collision_apis=True)
    # 物理マテリアルを明示的に指定しないと摩擦係数がPhysics側のデフォルト任せになり、
    # グリッパーが接触しても保持に足る摩擦力が得られず、掴んだ物体が滑り落ちることがある
    # (Isaac Simの公式トラブルシューティングでも摩擦係数の明示設定が案内されている)。
    grip_material = RigidBodyMaterial(
        "/World/physics_materials/grippy",
        static_frictions=[1.0],
        dynamic_frictions=[0.9],
    )
    target_geom.apply_physics_materials(grip_material)
    RigidPrim(paths=target_shape.paths)
    # フィンガー側の摩擦係数がFrankaアセットのデフォルトのままだと、target_object側だけ
    # 摩擦を上げても実効摩擦(組み合わせ)が不十分になり、持ち上げには成功しても
    # 高速な運搬中の慣性力で滑り落ちることがある(把持直後の複数回検証で確認済み)。
    finger_geom = GeomPrim(
        paths=["/World/panda/panda_leftfinger", "/World/panda/panda_rightfinger"],
        apply_collision_apis=True,
    )
    finger_geom.apply_physics_materials(grip_material)

    # Ref: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
    # Isaac Simのカメラは-Zをローカルの視線方向とする(world基準の+Zが上)ため、
    # target_object付近の真上にorientation指定なし(=単位クォータニオン)で置くだけで真下を向く。
    # 高さ0.6mだと、学習データ収集で物体をランダム配置できる範囲(画角に収まる範囲)が
    # 対角±7cm程度しかなく、yawも含めた姿勢の多様性を確保するには狭すぎる。
    # 0.8mまで上げて画角を広げ、対角±11cm程度まで拡張する(キューブの見かけサイズは
    # 139px→102pxに縮むが、8キーポイントのpose推定には十分な解像度)。
    camera = Camera(paths="/World/camera", positions=np.array([[0.5, 0.0, 0.8]]))
    # UsdGeom.Cameraのデフォルトの近クリップ面は1.0(stage単位=メートル)で、カメラ高さ0.8mより
    # 遠いため、workspace全体(0.5〜0.6m先)が近クリップ面の内側に入ってしまい何も描画されない。
    camera.set_clipping_ranges(near_distances=[0.01], far_distances=[10.0])

    build_action_graph(
        panda_prim_path="/World/panda",
        target_object_prim_path="/World/target_object",
        camera_prim_path="/World/camera",
    )

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
