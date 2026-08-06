#!/usr/bin/env python3
"""物体位置をランダム化した小規模合成データセットを/dataに生成する。

RGB画像に加えて、YOLO11-Pose学習用のキーポイントラベル・6D Pose(pose_gt.json)・
物体ローカル座標でのキーポイント定義(object_3d_keypoints.json)を出力する。
target_objectは自前でposeを設定して動かしているため、姿勢・キーポイントは
毎フレーム自分で計算した値をそのまま書き出す(BasicWriter/セマンティクスによる
自動検出には依存しない)。

SimulationAppはisaacsim配下の他モジュールをimportする前に必ず生成する必要があるため、
このスクリプトではargparse以外の一切のisaacsim/omni importをSimulationApp生成後に置く。
"""

import argparse
import os

from common import target_object_shape as shape

OBJECT_SIZE_M = shape.OBJECT_SIZE_M
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# カメラは(0.5, 0.0, 0.8)固定・真下向きで、実測画角(fx=fy=1527px, キューブ上面までの
# 距離0.75m)は約31cm x 24cm。yaw回転時のキューブ最大到達距離(対角の半分、約3.5cm)と
# 1cmの安全マージンを差し引いた、中心が確実に画角内に収まる範囲が以下。
# これより広い範囲でランダム配置すると、画角外でキーポイントが1つも見えないフレームが多発する。
SPAWN_X_RANGE = (0.388, 0.612)
SPAWN_Y_RANGE = (-0.073, 0.073)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="GUIを表示せずに起動する")
    parser.add_argument("--num-frames", type=int, default=50, help="生成するフレーム数")
    parser.add_argument("--output-dir", required=True, help="出力先ディレクトリ(data/<camera>/dataset/<version>)")
    return parser.parse_args()


def _quat_wxyz_from_yaw(yaw: float, np) -> list:
    half = yaw / 2.0
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _quat_wxyz_to_matrix(q, np):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    args = parse_args()

    from isaacsim import SimulationApp

    experience = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.full.kit")
    simulation_app = SimulationApp({"headless": args.headless}, experience=experience)

    import json

    import cv2
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.replicator.core", True)

    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import Camera, Cube
    from isaacsim.core.experimental.prims import GeomPrim
    from isaacsim.storage.native import get_assets_root_path

    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/ground",
    )

    target = Cube(
        paths="/World/target_object",
        positions=np.array([[0.5, 0.0, OBJECT_SIZE_M / 2.0]]),
        sizes=[OBJECT_SIZE_M],
        reset_xform_op_properties=True,
    )
    GeomPrim(paths=target.paths, apply_collision_apis=True)

    # Ref: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
    # run_simulation.py同様、カメラは-Zをローカルの視線方向とするため、
    # target_object付近の真上に単位クォータニオンで置くだけで真下を向く。カメラ自体は固定し、
    # target_objectの位置・yawだけをランダム化することでキーポイント計算をシンプルに保つ。
    # run_simulation.pyと同じ理由・同じ高さ(0.8m)で、SPAWN_X_RANGE/SPAWN_Y_RANGEの
    # 画角内に十分な余裕を持たせている。
    camera = Camera(paths="/World/camera", positions=np.array([[0.5, 0.0, 0.8]]))
    camera.set_clipping_ranges(near_distances=[0.01], far_distances=[10.0])
    rep.create.light(light_type="distant", intensity=1000.0)

    render_product = rep.create.render_product("/World/camera", (IMAGE_WIDTH, IMAGE_HEIGHT))
    rgb_anno = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_anno.attach(render_product)
    # _write_pose_label()のzcはUSDカメラ座標系から変換した平面Z(pinholeモデルのZ)なので、
    # 球面距離のdistance_to_cameraではなく、同じ規約のdistance_to_image_planeを使う。
    depth_anno = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_anno.attach(render_product)
    camparams_anno = rep.AnnotatorRegistry.get_annotator("CameraParams")
    camparams_anno.attach(render_product)

    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "object_3d_keypoints.json"), "w") as f:
        json.dump({"class": "target_object", "keypoints_local_xyz": shape.LOCAL_CORNERS}, f, indent=2)

    with open(os.path.join(args.output_dir, "dataset.yaml"), "w") as f:
        f.write(shape.dataset_yaml_content(os.path.abspath(args.output_dir)))

    rng = np.random.default_rng()
    pose_gt = []

    for i in range(args.num_frames):
        x = float(rng.uniform(*SPAWN_X_RANGE))
        y = float(rng.uniform(*SPAWN_Y_RANGE))
        yaw = float(rng.uniform(-np.pi, np.pi))
        orientation = _quat_wxyz_from_yaw(yaw, np)
        target.set_world_poses(
            positions=np.array([[x, y, OBJECT_SIZE_M / 2.0]]),
            orientations=np.array([orientation]),
        )

        # rep.orchestrator.step_async()はこの環境(pixi run経由・RTXレンダラー有効)だと
        # 初回呼び出しでCPU使用率100%のまま何十分経っても返ってこないハングを起こす
        # (asyncio側のイベントループとKitのSDGPipelineディスパッチの噛み合わせの問題と見られる)。
        # 同期版のstep()に置き換えると同条件で即座に完了するため、
        # asyncioを一切使わない構成にしている。
        rep.orchestrator.step(rt_subframes=4)

        rgb = rgb_anno.get_data()
        depth = depth_anno.get_data()
        cam = camparams_anno.get_data()

        frame_name = f"{i:04d}"
        cv2.imwrite(
            os.path.join(images_dir, f"{frame_name}.png"),
            cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR),
        )
        np.save(os.path.join(args.output_dir, f"depth_{frame_name}.npy"), depth)

        # get_world_poses()はwarp.arrayを返し、
        # numpy配列と違ってPython側で直接イテレーション・インデックスアクセスができないため、
        # 先にnumpyへ変換する。
        positions, orientations = target.get_world_poses()
        position = positions.numpy()[0]
        orientation_wxyz = orientations.numpy()[0]
        pose_gt.append(
            {
                "frame": frame_name,
                "position_xyz": [float(v) for v in position],
                "orientation_wxyz": [float(v) for v in orientation_wxyz],
            }
        )

        _write_pose_label(
            frame_name=frame_name,
            labels_dir=labels_dir,
            position=position,
            orientation_wxyz=orientation_wxyz,
            camera_params=cam,
            depth=depth,
            np=np,
        )

    with open(os.path.join(args.output_dir, "pose_gt.json"), "w") as f:
        json.dump(pose_gt, f, indent=2)

    simulation_app.close()


def _write_pose_label(*, frame_name, labels_dir, position, orientation_wxyz, camera_params, depth, np):
    # cameraViewTransformは実測でworld->camera(行ベクトル規約、USDカメラ座標系:+Y上,-Z前方)
    # であることを確認済み(ドキュメント上の「camera to world」という記述は誤り)。
    view_transform = np.array(camera_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    projection = np.array(camera_params["cameraProjection"], dtype=np.float64).reshape(4, 4)
    width, height = camera_params["renderProductResolution"]
    # fx/fyをcameraFocalLength+cameraApertureから再計算せず、
    # cameraProjectionから直接逆算する。
    # レンダー解像度とセンサーのアスペクト比が異なる場合に
    # Omniverse側が内部で行うaperture-fit調整をここで再現しなくて済むため。
    fx = projection[0, 0] * width / 2.0
    fy = projection[1, 1] * height / 2.0
    cx = width / 2.0
    cy = height / 2.0

    rotation = _quat_wxyz_to_matrix(orientation_wxyz, np)
    world_corners = np.array(shape.LOCAL_CORNERS) @ rotation.T + position
    world_corners = shape.canonicalize_corner_order(world_corners, rotation, np)

    # USDカメラ座標系(+Y上,-Z前方)からROS/pinhole慣例(+Y下,+Z前方)への変換は
    # ローカルX軸まわり180度回転(Y,Zの符号反転)に等しい。
    corners_homogeneous = np.concatenate([world_corners, np.ones((8, 1))], axis=1)
    points_camera_usd = corners_homogeneous @ view_transform
    camera_points = np.stack(
        [points_camera_usd[:, 0], -points_camera_usd[:, 1], -points_camera_usd[:, 2]], axis=1
    )

    us, vs, visible = shape.project_with_occlusion(camera_points, fx, fy, cx, cy, width, height, depth, np)
    if not any(flag > 0 for flag in visible):
        return
    shape.write_pose_label(os.path.join(labels_dir, f"{frame_name}.txt"), us, vs, visible, width, height, np)


if __name__ == "__main__":
    main()
