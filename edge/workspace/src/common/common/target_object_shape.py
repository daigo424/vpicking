#!/usr/bin/env python3
"""target_objectの形状・キーポイント定義と、YOLO-Poseラベル生成の共通処理。

cube_only.py/picking_session.pyの両方が全く同じキーポイントの生成順(1-indexed)・
対称性の畳み込みロジック・ラベル形式を使う必要があるため、ここに集約する
(重複させると、どちらか一方だけ修正して食い違う不具合を起こしやすい)。

isaacsim/omni importに関する制約(cube_only.py参照)と無関係な、mathとstdlibのみに
依存する軽量モジュールなので、SimulationApp生成前後どちらでimportしても問題ない。
numpyは呼び出し側からの引数として受け取り、このモジュール自身はimportしない
(cube_only.py側の「isaacsim/omni以外もSimulationApp生成後にまとめる」方針に合わせるため)。
"""

import math

OBJECT_SIZE_M = 0.05

# 対象物が地面(Z=0)にあると、joint2(panda_link0から0.333m上)への下向き到達距離が最大になる。
# 0.30mは実機のスタンド高さに近づける値だが、単独ではjoint2/6のトルク不足
# (下降が指令姿勢の数cm手前で止まる)を解消しない。
TABLE_HEIGHT_M = 0.30

# 直方体(立方体)の8頂点をオブジェクトローカル座標で定義する。同一平面上にない8点なので、
# 後続のPnP計算(cv2.solvePnP)で奥行き反転の曖昧性が出にくい構成になる。
_HALF = OBJECT_SIZE_M / 2.0
LOCAL_CORNERS = [
    [sx * _HALF, sy * _HALF, sz * _HALF] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]
# 上のLOCAL_CORNERSの生成順(1-indexed)に対応する、立方体の辺(12本)の接続関係。
# Ultralytics pose学習のdataset.yamlでスケルトン可視化に使う。
SKELETON = [
    [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4],
    [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8],
]

# 幾何学的に画角内へ投影できても、アームや物体自身に遮られて実際には見えないkeypointが
# 存在する(投影計算だけでは検出できない)。depth画像上の実測距離と投影計算上のzcを比較し、
# depth側が明らかに手前ならkeypointは遮蔽されているとみなす。値はセンサーノイズ・
# レンダリングの量子化誤差を吸収するための余裕。
OCCLUSION_MARGIN_M = 0.01


def rot_z(yaw: float, np):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def canonicalize_corner_order(world_corners, rotation, np):
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
    rotated_by_steps = local_corners_arr @ rot_z(steps * math.pi / 2, np).T
    perm = [
        int(np.argmin(np.sum((rotated_by_steps - local_corners_arr[j]) ** 2, axis=1)))
        for j in range(8)
    ]
    return world_corners[perm]


def project_with_occlusion(camera_points, fx, fy, cx, cy, width, height, depth_image, np):
    """カメラ座標系(pinhole慣例: +Z前方)の8点を画像へ投影し、depth画像との比較で遮蔽を判定する。

    戻り値は(us, vs, visible)。visibleは0=カメラ後方、1=画角外またはdepth比較で遮蔽、
    2=画角内かつ遮蔽なしで可視、を表す。
    """
    us, vs, visible = [], [], []
    for xc, yc, zc in camera_points:
        if zc <= 0:
            us.append(0.0)
            vs.append(0.0)
            visible.append(0)
            continue
        u = cx + fx * xc / zc
        v = cy + fy * yc / zc
        in_bounds = 0 <= u < width and 0 <= v < height
        # Ultralyticsのラベルローダーは正規化座標が[0,1]を外れるとラベルファイル全体を
        # 「破損」扱いにして学習から除外するため、画角外に出た点は画像端にクランプする
        # (可視性フラグでクランプ前の状態を区別する)。
        us.append(min(max(u, 0.0), width - 1))
        vs.append(min(max(v, 0.0), height - 1))
        if not in_bounds:
            visible.append(1)
            continue
        row = min(max(int(round(v)), 0), height - 1)
        col = min(max(int(round(u)), 0), width - 1)
        surface_depth = depth_image[row, col]
        occluded = np.isfinite(surface_depth) and surface_depth < zc - OCCLUSION_MARGIN_M
        visible.append(1 if occluded else 2)
    return us, vs, visible


def write_pose_label(label_path, us, vs, visible, width, height, np) -> None:
    us_arr, vs_arr = np.array(us), np.array(vs)
    visible_corners = [i for i, flag in enumerate(visible) if flag > 0]
    x_min, x_max = us_arr[visible_corners].min(), us_arr[visible_corners].max()
    y_min, y_max = vs_arr[visible_corners].min(), vs_arr[visible_corners].max()
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    bbox_w = (x_max - x_min) / width
    bbox_h = (y_max - y_min) / height

    fields = [0, x_center, y_center, bbox_w, bbox_h]
    for u, v, flag in zip(us, vs, visible):
        fields += [u / width, v / height, flag]

    with open(label_path, "w") as f:
        f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in fields) + "\n")


def dataset_yaml_content(output_dir_abspath: str) -> str:
    return (
        f"path: {output_dir_abspath}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: target_object\n"
        "kpt_shape: [8, 3]\n"
        "skeleton:\n" + "".join(f"  - {edge}\n" for edge in SKELETON)
    )
