#!/usr/bin/env python3
"""vp-run-gt実行中の実カメラ画像とGT姿勢から、YOLO11-Pose学習用データを記録する。

cube_only.pyの単純なキューブのみのシーンで学習すると、
実際のPandaロボットが写り込む本番シーンとの間にドメインギャップが生じ、
学習済みモデルがアームの一部をtarget_objectと誤検出する問題があった。
このスクリプトはgt_tf_publisher_nodeが配信する実際の物体姿勢と、
picking_controller_nodeの実動作でアームが写り込んだ実カメラ画像を組み合わせることで、
本番シーンと一致した学習データを作る。

事前にmake vp-run-gtでピッキングパイプラインを別途起動しておく必要がある
(gt_tf_publisher_node/camera_bridge_nodeが動いていないと画像・TFが得られない)。
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix

WORLD_FRAME = "world"
CAMERA_FRAME = "camera"
TARGET_FRAME = "target_object"
RGB_TOPIC = "/camera/rgb/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera_info"

OBJECT_SIZE_M = 0.05
# 幾何学的に画角内へ投影できても、実際にはアーム自身やキューブの反対側の面に
# 遮られて見えないkeypointが存在する(投影計算だけでは検出できない)。depth画像上の
# 実測距離と投影計算上のzcを比較し、depth側が明らかに手前ならkeypointは遮蔽されている
# とみなす。値はセンサーノイズ・レンダリングの量子化誤差を吸収するための余裕。
OCCLUSION_MARGIN_M = 0.01
_HALF = OBJECT_SIZE_M / 2.0
# cube_only.pyと全く同じ生成順(1-indexed)。学習用重みは1つのモデルとして使い回すため、
# キーポイントの意味(どの頂点が何番目か)を両スクリプトで揃えている。
LOCAL_CORNERS = [
    [sx * _HALF, sy * _HALF, sz * _HALF] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]
SKELETON = [
    [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4],
    [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8],
]
_LOCAL_CORNERS_ARR = np.array(LOCAL_CORNERS)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _canonicalize_corner_order(world_corners: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # cubeは90度Z回転ごとに見た目が同じになる(4回対称)。8頂点に固定の絶対番号を
    # 割り当てたまま学習すると、モデルは位置自体は正しく学習できても、
    # どの角が何番目かを一貫して取り違える(番号ズレ)ことが実測で確認できている。
    # yawの90度単位の成分をキーポイントの並び替えで吸収し、[-45,45)度に畳んだ
    # 端数のyawだけを見た目の違いとして残すことで、番号の意味を90度対称の範囲内で
    # 常に同じ相対的な角を指すよう揃える。
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    canonical_yaw = ((yaw + math.pi / 4) % (math.pi / 2)) - math.pi / 4
    steps = round((yaw - canonical_yaw) / (math.pi / 2))
    rotated_by_steps = _LOCAL_CORNERS_ARR @ _rot_z(steps * math.pi / 2).T
    perm = [
        int(np.argmin(np.sum((rotated_by_steps - _LOCAL_CORNERS_ARR[j]) ** 2, axis=1)))
        for j in range(8)
    ]
    return world_corners[perm]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data/dataset")
    parser.add_argument("--max-frames", type=int, default=60, help="このプロセスで記録する最大フレーム数")
    parser.add_argument("--min-interval-sec", type=float, default=0.15, help="フレーム間の最小記録間隔")
    parser.add_argument("--start-index", type=int, default=0, help="ファイル名の開始インデックス")
    return parser.parse_args()


def _transform_to_matrix(translation, rotation) -> np.ndarray:
    mat = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
    mat[0, 3] = translation.x
    mat[1, 3] = translation.y
    mat[2, 3] = translation.z
    return mat


class RecorderNode(Node):
    def __init__(self, args) -> None:
        super().__init__("picking_session")
        self._args = args
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._depth_image: np.ndarray | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._frame_count = 0
        self._last_saved_time = 0.0

        self._images_dir = os.path.join(args.output_dir, "images")
        self._labels_dir = os.path.join(args.output_dir, "labels")
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._labels_dir, exist_ok=True)
        self._pose_gt_path = os.path.join(args.output_dir, "pose_gt.json")
        # 複数回の短時間実行を積み重ねてデータセットを作る運用では、1回の実行が
        # max_framesに到達せずSIGKILLで終了することが多く、終了時にまとめて書き出す設計だと
        # dataset.yaml/object_3d_keypoints.jsonが一度も生成されないまま終わる。
        # データに依存しない静的ファイルは起動時に、pose_gt.jsonはフレームごとに即座に書き出す。
        self._write_static_outputs()

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)
        self.create_subscription(Image, RGB_TOPIC, self._on_rgb, 10)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_depth(self, msg: Image) -> None:
        self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_rgb(self, msg: Image) -> None:
        if self._camera_info is None or self._depth_image is None or self._frame_count >= self._args.max_frames:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_saved_time < self._args.min_interval_sec:
            return

        try:
            world_to_object = self._tf_buffer.lookup_transform(WORLD_FRAME, TARGET_FRAME, rclpy.time.Time())
            world_to_camera = self._tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        frame_name = f"{self._args.start_index + self._frame_count:04d}"
        wrote = self._write_label(frame_name, world_to_object.transform, world_to_camera.transform)
        if not wrote:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(os.path.join(self._images_dir, f"{frame_name}.png"), frame)

        t = world_to_object.transform.translation
        r = world_to_object.transform.rotation
        self._append_pose_gt(
            {"frame": frame_name, "position_xyz": [t.x, t.y, t.z], "orientation_xyzw": [r.x, r.y, r.z, r.w]}
        )

        self._frame_count += 1
        self._last_saved_time = now
        self.get_logger().info(f"frame {frame_name} recorded ({self._frame_count}/{self._args.max_frames})")

        if self._frame_count >= self._args.max_frames:
            raise SystemExit(0)

    def _write_label(self, frame_name, object_transform, camera_transform) -> bool:
        fx, fy = self._camera_info.k[0], self._camera_info.k[4]
        cx, cy = self._camera_info.k[2], self._camera_info.k[5]
        width, height = self._camera_info.width, self._camera_info.height
        depth_image = self._depth_image

        world_to_camera_mat = _transform_to_matrix(camera_transform.translation, camera_transform.rotation)
        camera_to_world_mat = np.linalg.inv(world_to_camera_mat)

        rotation = quaternion_matrix(
            [object_transform.rotation.x, object_transform.rotation.y, object_transform.rotation.z, object_transform.rotation.w]
        )[:3, :3]
        position = np.array([object_transform.translation.x, object_transform.translation.y, object_transform.translation.z])
        world_corners = np.array(LOCAL_CORNERS) @ rotation.T + position
        world_corners = _canonicalize_corner_order(world_corners, rotation)

        us, vs, visible = [], [], []
        for corner in world_corners:
            point_camera = camera_to_world_mat @ np.array([corner[0], corner[1], corner[2], 1.0])
            xc, yc, zc = point_camera[:3]
            if zc <= 0:
                us.append(0.0)
                vs.append(0.0)
                visible.append(0)
                continue
            u = cx + fx * xc / zc
            v = cy + fy * yc / zc
            in_bounds = 0 <= u < width and 0 <= v < height
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

        us_arr, vs_arr = np.array(us), np.array(vs)
        visible_corners = [i for i, flag in enumerate(visible) if flag > 0]
        # 画角内に十分な頂点が写っていないフレーム(アームに完全遮蔽・画角外)は
        # 学習データとして使えないため記録しない。
        if len([i for i in visible_corners if visible[i] == 2]) < 4:
            return False

        x_min, x_max = us_arr[visible_corners].min(), us_arr[visible_corners].max()
        y_min, y_max = vs_arr[visible_corners].min(), vs_arr[visible_corners].max()
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        bbox_w = (x_max - x_min) / width
        bbox_h = (y_max - y_min) / height

        fields = [0, x_center, y_center, bbox_w, bbox_h]
        for u, v, flag in zip(us, vs, visible):
            fields += [u / width, v / height, flag]

        with open(os.path.join(self._labels_dir, f"{frame_name}.txt"), "w") as f:
            f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in fields) + "\n")
        return True

    def _append_pose_gt(self, entry: dict) -> None:
        existing = []
        if os.path.exists(self._pose_gt_path):
            with open(self._pose_gt_path) as f:
                existing = json.load(f)
        with open(self._pose_gt_path, "w") as f:
            json.dump(existing + [entry], f, indent=2)

    def _write_static_outputs(self) -> None:
        keypoints_path = os.path.join(self._args.output_dir, "object_3d_keypoints.json")
        if not os.path.exists(keypoints_path):
            with open(keypoints_path, "w") as f:
                json.dump({"class": "target_object", "keypoints_local_xyz": LOCAL_CORNERS}, f, indent=2)

        yaml_path = os.path.join(self._args.output_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            with open(yaml_path, "w") as f:
                f.write(
                    f"path: {os.path.abspath(self._args.output_dir)}\n"
                    "train: images\n"
                    "val: images\n"
                    "names:\n"
                    "  0: target_object\n"
                    "kpt_shape: [8, 3]\n"
                    "skeleton:\n" + "".join(f"  - {edge}\n" for edge in SKELETON)
                )


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RecorderNode(args)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
