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

CAMERA_NAMESPACE環境変数で、俯瞰カメラ(既定値"camera")・手先カメラ("wrist_camera"等)の
どちらのデータを収集するかを切り替えられる(pose_estimation_node.pyと同じ命名規則)。
"""

import argparse
import os

import rclpy
from cv_bridge import CvBridge
from frame_recording import FrameWriter
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

WORLD_FRAME = "world"
CAMERA_NAMESPACE = os.environ.get("CAMERA_NAMESPACE", "camera")
CAMERA_FRAME = CAMERA_NAMESPACE
TARGET_FRAME = "target_object"
RGB_TOPIC = f"/{CAMERA_NAMESPACE}/rgb/image_raw"
DEPTH_TOPIC = f"/{CAMERA_NAMESPACE}/depth/image_raw"
CAMERA_INFO_TOPIC = f"/{CAMERA_NAMESPACE}/camera_info"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="data/<camera>/dataset/<version>のパス")
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
        self._depth_image = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._frame_count = 0
        self._last_saved_time = 0.0
        # 複数回の短時間実行を積み重ねてデータセットを作る運用では、1回の実行が
        # max_framesに到達せずSIGKILLで終了することが多く、終了時にまとめて書き出す設計だと
        # dataset.yaml/object_3d_keypoints.jsonが一度も生成されないまま終わる。FrameWriterは
        # 静的ファイルをコンストラクタで即座に書き出すため、この用途にそのまま使える。
        self._writer = FrameWriter(args.output_dir)

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
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        wrote = self._writer.try_write_frame(
            frame_name, frame, self._depth_image, self._camera_info, world_to_object.transform, world_to_camera.transform
        )
        if not wrote:
            return

        self._frame_count += 1
        self._last_saved_time = now
        self.get_logger().info(f"frame {frame_name} recorded ({self._frame_count}/{self._args.max_frames})")

        if self._frame_count >= self._args.max_frames:
            raise SystemExit(0)


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
