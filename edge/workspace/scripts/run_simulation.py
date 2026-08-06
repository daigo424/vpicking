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


def build_action_graph(panda_prim_path: str, target_object_prim_paths: list[str], cameras: list[dict]):
    """ROS2Context / JointState Pub-Sub / ArticulationController / 物体GT位置姿勢配線 / カメラ配線を構築する。

    Ref: https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/isaac_panda/launch/isaac_moveit.py
    関節のコマンド購読・状態配信はIsaacArticulationController/IsaacReadJointState
    (OmniGraph標準ノード、MoveIt公式のIsaac Sim連携チュートリアルが実際に使っている構成)を使う。
    以前はNVIDIAフォーラムの1件の投稿(ノードグラフの評価タイミングに起因する軌道追従の
    不安定さの報告)を根拠に、Frankaのarticulation API(set_dof_position_targets/
    get_dof_positions)を直接叩く自前のrclpyノードに置き換えていたが、公式チュートリアルが
    標準ノードで実際に動作確認していることを優先し、標準構成に戻す。

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
      - width, height: レンダリング解像度(px)。画角を広げたカメラでは、同じ解像度のままだと
        物体の見かけサイズ(px)が縮んで検出精度が落ちるため、カメラごとに変えられるようにする。
    """
    import omni.graph.core as og

    # Ref: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/migration_guides/isaac_sim_6_0/ros2_omnigraph_migration.html
    # Isaac Sim 6.0でROS2PublishTransformTreeにtargetPrim(s)を直接渡す方式が非推奨になり、
    # IsaacComputeTransformTreeが計算した値を明示的に渡す構成が推奨されているため、
    # その配線に従っている。
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
        ("ROS2PublishJointState.inputs:topicName", "/isaac_joint_states"),
        ("ROS2SubscribeJointState.inputs:topicName", "/isaac_joint_commands"),
        (
            "ComputeGtTransformTree.inputs:targetPrims",
            target_object_prim_paths + [camera["prim_path"] for camera in cameras],
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
            (f"{render_node}.inputs:width", camera["width"]),
            (f"{render_node}.inputs:height", camera["height"]),
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
    from common.target_object_shape import TABLE_HEIGHT_M
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

    # target_objectをTABLE_HEIGHT_M(common.target_object_shape参照)の高さで支えるための
    # 静的な作業台。ワークスペース(デフォルト位置0.5,0.0、プレイス位置0.3,0.3、
    # STACK_BLOCKS配置範囲)を覆う大きさにしている。RigidPrimを付けない(=静的)ことで
    # 重力の影響を受けず、常に同じ高さで対象物を支え続ける。
    table = Cube(
        paths="/World/table",
        positions=np.array([[0.5, 0.0, TABLE_HEIGHT_M / 2.0]]),
        sizes=[1.0],
        scales=np.array([[0.7, 0.8, TABLE_HEIGHT_M]]),
        colors="#8a7048",
        reset_xform_op_properties=True,
    )
    GeomPrim(paths=table.paths, apply_collision_apis=True)

    # 物理マテリアルを明示的に指定しないと摩擦係数がPhysics側のデフォルト任せになり、
    # グリッパーが接触しても保持に足る摩擦力が得られず、掴んだ物体が滑り落ちることがある
    # (Isaac Simの公式トラブルシューティングでも摩擦係数の明示設定が案内されている)。
    grip_material = RigidBodyMaterial(
        "/World/physics_materials/grippy",
        static_frictions=[1.0],
        dynamic_frictions=[0.9],
    )

    def spawn_cube(path: str, x: float, y: float, yaw: float):
        orientation = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        shape = Cube(
            paths=path,
            positions=np.array([[x, y, TABLE_HEIGHT_M + 0.05]]),
            orientations=np.array([orientation]),
            sizes=[0.05],
            reset_xform_op_properties=True,
        )
        geom = GeomPrim(paths=shape.paths, apply_collision_apis=True)
        geom.apply_physics_materials(grip_material)
        RigidPrim(paths=shape.paths)
        return shape

    # カメラ高さ・画角は物体配置範囲の計算(画角に収まる安全範囲)でも使うため、
    # カメラ自体の生成より先に決めておく(実際のCamera生成は後段、この値をそのまま使う)。
    # 画角はaperture/focal_lengthのpinholeモデルで決まり、距離によらず一定の値になる。
    # 俯瞰カメラの既定画角(23.6度、focal_length=5.0mm)はアーム周辺に散らすには狭すぎるため、
    # focal_lengthを半分にして画角を約2倍に広げる(下のCamera生成時に適用)。
    # apertureの値はIsaac Simのカメラ既定値(2.0955mm x 1.52908mm、変更していない)。
    camera_height_m = float(os.environ.get("CAMERA_HEIGHT_M", 0.8))
    overhead_focal_length_mm = 2.5
    overhead_aperture_x_mm, overhead_aperture_y_mm = 2.0955, 1.52908

    # STACK_BLOCKS環境変数(3個スタッキングデモ用)が指定されていれば、ランダム配置した
    # 複数キューブ(/World/stack_block_<N>)を、指定なければ従来通り単一のtarget_objectを生成する。
    # target_objectを前提にした既存の学習データ収集スクリプト・認識ノードの挙動を変えないため、
    # 未指定時の構成(prim名・座標)は変更しない。
    num_stack_blocks = int(os.environ.get("STACK_BLOCKS", "0"))
    if num_stack_blocks > 0:
        # カメラからキューブ上面までの実効距離(キューブ上面はテーブルから5cm)に対して、
        # pinholeモデルで画角内に収まる範囲を計算し、yaw回転時のキューブ最大到達距離
        # (対角の半分、約3.5cm)と1cmの安全マージンを差し引く。検出モデルが単一距離・
        # 単一画角でしか学習されていなかった間はこの範囲を広げる余地がなかったが、
        # 複数距離を学習データに含めることでその制約は解消したため、CAMERA_HEIGHT_Mや
        # 画角を変えれば配置範囲もそれに応じて広がる。
        cube_top_offset_m = 0.05
        diagonal_margin_m = 0.06
        distance_m = camera_height_m - TABLE_HEIGHT_M - cube_top_offset_m
        half_x = distance_m * overhead_aperture_x_mm / (2 * overhead_focal_length_mm) - diagonal_margin_m
        half_y = distance_m * overhead_aperture_y_mm / (2 * overhead_focal_length_mm) - diagonal_margin_m
        # 画角上は見えていても、アームの土台に近すぎる位置は真下向き姿勢での到達性が悪く、
        # 把持・運搬中に物体を落としやすい(検出精度とは別の、腕の可動域由来の制約)。
        # 視認性ベースの範囲とは独立に、x方向はこの下限で切り詰める。
        arm_min_x_m = 0.30
        x_min, x_max = max(0.5 - half_x, arm_min_x_m), 0.5 + half_x
        y_min, y_max = -half_y, half_y
        min_separation_m = 0.09
        positions_xy: list[tuple[float, float]] = []
        for _ in range(num_stack_blocks):
            for _attempt in range(200):
                x = float(np.random.uniform(x_min, x_max))
                y = float(np.random.uniform(y_min, y_max))
                if all(np.hypot(x - px, y - py) >= min_separation_m for px, py in positions_xy):
                    positions_xy.append((x, y))
                    break
            else:
                positions_xy.append((x, y))
        target_paths = []
        for i, (x, y) in enumerate(positions_xy, start=1):
            yaw = float(np.random.uniform(-np.pi, np.pi))
            path = f"/World/stack_block_{i}"
            spawn_cube(path, x, y, yaw)
            target_paths.append(path)
    else:
        # TARGET_OBJECT_X/Y/YAW環境変数が指定されていればそれを使う。学習データ収集用に
        # 物体の初期位置・向きを実行のたびに変えられるようにするためのフックで、未指定時は
        # 従来通り固定位置・向きになる。
        target_x = float(os.environ.get("TARGET_OBJECT_X", 0.5))
        target_y = float(os.environ.get("TARGET_OBJECT_Y", 0.0))
        target_yaw = float(os.environ.get("TARGET_OBJECT_YAW", 0.0))
        spawn_cube("/World/target_object", target_x, target_y, target_yaw)
        target_paths = ["/World/target_object"]
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
    # 対角±7cm程度しかなく、yawも含めた姿勢の多様性を確保するには狭すぎるため、
    # 0.8mを基準の高さとする。
    # 検出モデルは学習データに含まれる距離での見かけサイズにしか対応できないため、
    # CAMERA_HEIGHT_M環境変数で複数の距離を学習データ・実行時の両方に持たせられるようにする
    # (値自体は物体配置範囲の計算のため既に上で読み込み済み。未指定時は従来通り0.8m固定)。
    camera = Camera(paths="/World/camera", positions=np.array([[0.5, 0.0, camera_height_m]]))
    # 画角を広げた分(overhead_focal_length_mm、既定50mmの半分)、同じ解像度のままだと
    # キューブの見かけサイズが縮んで検出精度が落ちるため、解像度も640x480から960x720へ上げ、
    # フレーム内で使える実質的なpx数を確保する(build_action_graph呼び出し側のcameras[]で指定)。
    camera.set_focal_lengths(focal_lengths=[overhead_focal_length_mm])
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
        target_object_prim_paths=target_paths,
        cameras=[
            {
                "prim_path": "/World/camera",
                "node_prefix": "overhead",
                "topic_prefix": "/sim_camera",
                "frame_id": "camera",
                "width": 960,
                "height": 720,
            },
            {
                "prim_path": "/World/panda/panda_hand/wrist_camera",
                "node_prefix": "wrist",
                "topic_prefix": "/sim_wrist_camera",
                "frame_id": "wrist_camera",
                "width": 640,
                "height": 480,
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
