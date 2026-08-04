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


def build_action_graph(panda_prim_path: str, target_object_prim_path: str, cameras: list[dict]):
    """ROS2Context / JointState Pub-Sub / ArticulationController / 物体GT位置姿勢配線 / カメラ配線を構築する。

    target_objectとカメラの真値TFは最終的な`/tf`ではなく`/ground_truth/tf`にpublishする。
    `vision_picking`パッケージの`gt_tf_publisher_node`がこれをsubscribeして
    world -> target_objectとして`/tf`に再publishする構成にすることで、
    認識方式を変える場合でも`/tf`へのpublisher側を差し替えるだけで済むようにするため。
    カメラ画像も同様に、OmniGraph側の実装依存のトピック名で一旦publishし、
    認識ノード側が依存する安定したトピック名への変換は`vision_picking`パッケージ側の
    ノードに委ねる。

    camerasは俯瞰・手先カメラ等、複数台分の設定を表す辞書のリスト。各要素のキー:
      - prim_path: カメラプリムのパス
      - node_prefix: このカメラ用に生成するOmniGraphノード名の接頭辞(カメラ間で重複しないこと)
      - topic_prefix: 中継トピック名の接頭辞({topic_prefix}/rgb 等)
      - frame_id: TFのframe_id(camera_bridge_node側で完全一致判定に使う)
    """
    import omni.graph.core as og

    # Ref: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/migration_guides/isaac_sim_6_0/ros2_omnigraph_migration.html
    # Isaac Sim 6.0でROS2PublishTransformTree/ROS2PublishJointStateにtargetPrim(s)を直接渡す方式が
    # 非推奨になり、IsaacComputeTransformTree/IsaacReadJointStateが計算した値を明示的に渡す構成が
    # 推奨されているため、その配線に従っている。
    create_nodes = [
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
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ROS2SubscribeJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ComputeGtTransformTree.inputs:execIn"),
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
        ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
        ("ROS2Context.outputs:context", "ROS2PublishJointState.inputs:context"),
        ("ROS2Context.outputs:context", "ROS2SubscribeJointState.inputs:context"),
        ("ROS2Context.outputs:context", "ROS2PublishGtTransformTree.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "ROS2PublishGtTransformTree.inputs:timeStamp"),
        ("ROS2SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
        ("ROS2SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
        ("ROS2SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
        ("ROS2SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
    ]
    set_values = [
        ("ReadJointState.inputs:prim", panda_prim_path),
        ("ArticulationController.inputs:targetPrim", panda_prim_path),
        ("ROS2PublishJointState.inputs:topicName", "/joint_states"),
        ("ROS2SubscribeJointState.inputs:topicName", "/joint_command"),
        (
            "ComputeGtTransformTree.inputs:targetPrims",
            [target_object_prim_path] + [camera["prim_path"] for camera in cameras],
        ),
        ("ROS2PublishGtTransformTree.inputs:topicName", "/ground_truth/tf"),
    ]

    for camera in cameras:
        prefix = camera["node_prefix"]
        render_node = f"CreateCameraRenderProduct_{prefix}"
        rgb_node = f"ROS2PublishCameraRgb_{prefix}"
        depth_node = f"ROS2PublishCameraDepth_{prefix}"
        info_node = f"ROS2PublishCameraInfo_{prefix}"
        create_nodes += [
            (render_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (rgb_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        connect += [
            ("OnPlaybackTick.outputs:tick", f"{render_node}.inputs:execIn"),
            (f"{render_node}.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            (f"{render_node}.outputs:execOut", f"{depth_node}.inputs:execIn"),
            (f"{render_node}.outputs:execOut", f"{info_node}.inputs:execIn"),
            (f"{render_node}.outputs:renderProductPath", f"{rgb_node}.inputs:renderProductPath"),
            (f"{render_node}.outputs:renderProductPath", f"{depth_node}.inputs:renderProductPath"),
            (f"{render_node}.outputs:renderProductPath", f"{info_node}.inputs:renderProductPath"),
            ("ROS2Context.outputs:context", f"{rgb_node}.inputs:context"),
            ("ROS2Context.outputs:context", f"{depth_node}.inputs:context"),
            ("ROS2Context.outputs:context", f"{info_node}.inputs:context"),
        ]
        set_values += [
            (f"{render_node}.inputs:cameraPrim", camera["prim_path"]),
            (f"{render_node}.inputs:width", 640),
            (f"{render_node}.inputs:height", 480),
            (f"{rgb_node}.inputs:type", "rgb"),
            (f"{rgb_node}.inputs:topicName", f"{camera['topic_prefix']}/rgb"),
            (f"{rgb_node}.inputs:frameId", camera["frame_id"]),
            (f"{depth_node}.inputs:type", "depth"),
            (f"{depth_node}.inputs:topicName", f"{camera['topic_prefix']}/depth"),
            (f"{depth_node}.inputs:frameId", camera["frame_id"]),
            (f"{info_node}.inputs:topicName", f"{camera['topic_prefix']}/camera_info"),
            (f"{info_node}.inputs:frameId", camera["frame_id"]),
        ]

    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connect,
            og.Controller.Keys.SET_VALUES: set_values,
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

    # 俯瞰カメラは奥行き信号が弱く、アームがカメラと物体の間を横切ると遮蔽されたキーポイントを
    # 誤って高信頼度で出力し、姿勢推定が破綻することがある。手先(panda_hand)に追従する
    # 第2のカメラを追加し、俯瞰カメラでの大まかな位置把握 -> 手先カメラでの近距離での
    # 精緻化という2段階の姿勢推定を可能にする。
    # translations/orientationsはpanda_handのローカル座標系での指定(positions/orientationsは
    # world座標系になるため、親プリムに追従させるにはこちらを使う必要がある)。
    #
    # panda_handの中心軸(ローカルZ)はグリッパー自身の指が伸びる軸と重なるため、そこにオフセット
    # して真下を向かせると指の表面しか映らない。UsdGeom.BBoxCacheで計測したpanda_hand/
    # panda_leftfinger/panda_rightfingerのローカル座標系での範囲では、指の開閉軸はローカルYで
    # (指はY=±0.04・X=±0.025・Z=0.06〜0.11)、hand本体はZ<0.066までしか無い。そのため
    # ローカルX方向(開閉軸と直交・hand本体の外)へオフセットし、Zは指の付け根より少し先
    # (0.08、hand本体の外・指先より手前)に置くことで、hand本体・指のどちらとも干渉しない
    # 位置を確保している。向きは素直な真下(panda_handのローカルZ+方向)のまま。
    # 俯瞰カメラの既定画角(23.6度)は近距離でのオフセットに対する許容度が低すぎるため、
    # 手先カメラだけ焦点距離を短くして画角を広げ、位置ずれに対する頑健性を確保する。
    wrist_camera = Camera(
        paths="/World/panda/panda_hand/wrist_camera",
        translations=np.array([[0.05, 0.0, 0.08]]),
        orientations=np.array([[0.0, 1.0, 0.0, 0.0]]),
    )
    wrist_camera.set_clipping_ranges(near_distances=[0.01], far_distances=[10.0])
    wrist_camera.set_focal_lengths(focal_lengths=[1.2])

    build_action_graph(
        panda_prim_path="/World/panda",
        target_object_prim_path="/World/target_object",
        cameras=[
            {
                "prim_path": "/World/camera",
                "node_prefix": "overhead",
                "topic_prefix": "/sim_camera",
                "frame_id": "camera",
            },
            {
                "prim_path": "/World/panda/panda_hand/wrist_camera",
                "node_prefix": "wrist",
                "topic_prefix": "/sim_wrist_camera",
                "frame_id": "wrist_camera",
            },
        ],
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
