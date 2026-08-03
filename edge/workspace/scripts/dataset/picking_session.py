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
import os

import cv2
import numpy as np
import rclpy
from common import target_object_shape as shape
from common.camera_intrinsics import intrinsics_from_camera_info
from common.transforms import transform_to_matrix
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

# 学習データとして使うには、visible=2(遮蔽なし・画角内)の頂点が最低これだけ必要。
# 画角内に十分な頂点が写っていないフレーム(アームに完全遮蔽・画角外)は記録しない。
MIN_VISIBLE_CORNERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data/dataset")
    parser.add_argument("--max-frames", type=int, default=60, help="このプロセスで記録する最大フレーム数")
    parser.add_argument("--min-interval-sec", type=float, default=0.15, help="フレーム間の最小記録間隔")
    parser.add_argument("--start-index", type=int, default=0, help="ファイル名の開始インデックス")
    return parser.parse_args()


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
        fx, fy, cx, cy = intrinsics_from_camera_info(self._camera_info)
        width, height = self._camera_info.width, self._camera_info.height

        world_to_camera_mat = transform_to_matrix(camera_transform)
        camera_to_world_mat = np.linalg.inv(world_to_camera_mat)

        rotation = quaternion_matrix(
            [object_transform.rotation.x, object_transform.rotation.y, object_transform.rotation.z, object_transform.rotation.w]
        )[:3, :3]
        position = np.array([object_transform.translation.x, object_transform.translation.y, object_transform.translation.z])
        world_corners = np.array(shape.LOCAL_CORNERS) @ rotation.T + position
        world_corners = shape.canonicalize_corner_order(world_corners, rotation, np)

        corners_homogeneous = np.concatenate([world_corners, np.ones((8, 1))], axis=1)
        camera_points = (corners_homogeneous @ camera_to_world_mat.T)[:, :3]
        us, vs, visible = shape.project_with_occlusion(
            camera_points, fx, fy, cx, cy, width, height, self._depth_image, np
        )

        if sum(1 for flag in visible if flag == 2) < MIN_VISIBLE_CORNERS:
            return False

        shape.write_pose_label(os.path.join(self._labels_dir, f"{frame_name}.txt"), us, vs, visible, width, height, np)
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
                json.dump({"class": "target_object", "keypoints_local_xyz": shape.LOCAL_CORNERS}, f, indent=2)

        yaml_path = os.path.join(self._args.output_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            with open(yaml_path, "w") as f:
                f.write(shape.dataset_yaml_content(os.path.abspath(self._args.output_dir)))


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
