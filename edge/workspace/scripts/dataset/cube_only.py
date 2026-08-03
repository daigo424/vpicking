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
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="GUIを表示せずに起動する")
    parser.add_argument("--num-frames", type=int, default=50, help="生成するフレーム数")
    parser.add_argument("--output-dir", default="/data/dataset", help="出力先ディレクトリ")
    return parser.parse_args()


OBJECT_SIZE_M = 0.05
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# カメラは(0.5, 0.0, 0.8)固定・真下向きで、実測画角(fx=fy=1527px, キューブ上面までの
# 距離0.75m)は約31cm x 24cm。yaw回転時のキューブ最大到達距離(対角の半分、約3.5cm)と
# 1cmの安全マージンを差し引いた、中心が確実に画角内に収まる範囲が以下。
# これより広い範囲でランダム配置すると、画角外でキーポイントが1つも見えないフレームが多発する。
SPAWN_X_RANGE = (0.388, 0.612)
SPAWN_Y_RANGE = (-0.073, 0.073)

# 直方体(立方体)の8頂点をオブジェクトローカル座標で定義する。同一平面上にない8点なので、
# 後続のPnP計算(cv2.solvePnP)で奥行き反転の曖昧性が出にくい構成になる。
_HALF = OBJECT_SIZE_M / 2.0
LOCAL_CORNERS = [
    [sx * _HALF, sy * _HALF, sz * _HALF] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]
# 幾何学的に画角内へ投影できても、真上のカメラからだとキューブ自身の胴体に遮られて
# 実際には見えない下面側のkeypointが存在する(投影計算だけでは検出できない)。depth画像上の
# 実測距離と投影計算上のzcを比較し、depth側が明らかに手前ならkeypointは遮蔽されている
# とみなす。値はセンサーノイズ・レンダリングの量子化誤差を吸収するための余裕。
OCCLUSION_MARGIN_M = 0.01
# 上のLOCAL_CORNERSの生成順(1-indexed)に対応する、立方体の辺(12本)の接続関係。
# Ultralytics pose学習のdataset.yamlでスケルトン可視化に使う。
SKELETON = [
    [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4],
    [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8],
]


def _quat_wxyz_from_yaw(yaw: float, np) -> list:
    half = yaw / 2.0
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _rot_z(yaw: float, np):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _canonicalize_corner_order(world_corners, rotation, np):
    # cubeは90度Z回転ごとに見た目が同じになる(4回対称)。8頂点に固定の絶対番号を
    # 割り当てたまま学習すると、モデルは位置自体は正しく学習できても、
    # どの角が何番目かを一貫して取り違える(番号ズレ)ことが実測で確認できている。
    # yawの90度単位の成分をキーポイントの並び替えで吸収し、[-45,45)度に畳んだ
    # 端数のyawだけを見た目の違いとして残すことで、番号の意味を90度対称の範囲内で
    # 常に同じ相対的な角を指すよう揃える。
    local_corners_arr = np.array(LOCAL_CORNERS)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    canonical_yaw = ((yaw + math.pi / 4) % (math.pi / 2)) - math.pi / 4
    steps = round((yaw - canonical_yaw) / (math.pi / 2))
    rotated_by_steps = local_corners_arr @ _rot_z(steps * math.pi / 2, np).T
    perm = [
        int(np.argmin(np.sum((rotated_by_steps - local_corners_arr[j]) ** 2, axis=1)))
        for j in range(8)
    ]
    return world_corners[perm]


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
        json.dump({"class": "target_object", "keypoints_local_xyz": LOCAL_CORNERS}, f, indent=2)

    with open(os.path.join(args.output_dir, "dataset.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(args.output_dir)}\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: target_object\n"
            "kpt_shape: [8, 3]\n"
            "skeleton:\n" + "".join(f"  - {edge}\n" for edge in SKELETON)
        )

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
    world_corners = np.array(LOCAL_CORNERS) @ rotation.T + position
    world_corners = _canonicalize_corner_order(world_corners, rotation, np)

    us, vs, visible = [], [], []
    for corner in world_corners:
        point_world_row = np.array([corner[0], corner[1], corner[2], 1.0])
        point_camera_usd = point_world_row @ view_transform
        # USDカメラ座標系(+Y上,-Z前方)からROS/pinhole慣例(+Y下,+Z前方)への変換は
        # ローカルX軸まわり180度回転(Y,Zの符号反転)に等しい。
        xc = point_camera_usd[0]
        yc = -point_camera_usd[1]
        zc = -point_camera_usd[2]
        if zc <= 0:
            us.append(0.0)
            vs.append(0.0)
            visible.append(0)
            continue
        u = cx + fx * xc / zc
        v = cy + fy * yc / zc
        in_bounds = 0 <= u < width and 0 <= v < height
        # Ultralyticsのラベルローダーは正規化座標が[0,1]を外れると
        # ラベルファイル全体を「破損」扱いにして学習から除外するため、
        # 画角外に出た点は画像端にクランプする(可視性フラグでクランプ前の状態を区別する)。
        us.append(min(max(u, 0.0), width - 1))
        vs.append(min(max(v, 0.0), height - 1))
        if not in_bounds:
            visible.append(1)
            continue
        row = min(max(int(round(v)), 0), height - 1)
        col = min(max(int(round(u)), 0), width - 1)
        surface_depth = depth[row, col]
        occluded = np.isfinite(surface_depth) and surface_depth < zc - OCCLUSION_MARGIN_M
        visible.append(1 if occluded else 2)

    us_arr, vs_arr = np.array(us), np.array(vs)
    visible_corners = [i for i, flag in enumerate(visible) if flag > 0]
    if not visible_corners:
        return
    x_min, x_max = us_arr[visible_corners].min(), us_arr[visible_corners].max()
    y_min, y_max = vs_arr[visible_corners].min(), vs_arr[visible_corners].max()
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    bbox_w = (x_max - x_min) / width
    bbox_h = (y_max - y_min) / height

    fields = [0, x_center, y_center, bbox_w, bbox_h]
    for u, v, flag in zip(us, vs, visible):
        fields += [u / width, v / height, flag]

    with open(os.path.join(labels_dir, f"{frame_name}.txt"), "w") as f:
        f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in fields) + "\n")


if __name__ == "__main__":
    main()
