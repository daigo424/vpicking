#!/usr/bin/env python3
"""手先カメラの学習データを、アームを物体近傍のランダムな相対位置・向きへ動かしながら収集する。

picking_session.py(vp-run-gt実行中の自然なアプローチ軌道をそのまま記録する)だけでは、
手先カメラがpanda_handからローカルX+0.05mオフセットして取り付けられている都合上、
グリッパーを物体中心に正しく合わせるほど物体は画像内の同じような位置(実測ではlabels.jpgで
確認した通り左上寄り)にしか写らないデータに偏る。このスクリプトはground truthを使って
アームを物体近傍のランダムな相対位置・向きへ直接移動させることで、物体が画像内の
様々な位置・大きさで写るデータを収集する。

事前にmake vp-run-gt相当(gt_tf_publisher_node/camera_bridge_node)を別途起動しておく必要がある。
"""

import argparse
import math
import os
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from frame_recording import FrameWriter
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from vision_picking.picking_robot_interface import RobotInterface, WORLD_FRAME, downward_orientation_for_yaw

TARGET_FRAME = "target_object"
CAMERA_NAMESPACE = os.environ.get("CAMERA_NAMESPACE", "wrist_camera")
CAMERA_FRAME = CAMERA_NAMESPACE
RGB_TOPIC = f"/{CAMERA_NAMESPACE}/rgb/image_raw"
DEPTH_TOPIC = f"/{CAMERA_NAMESPACE}/depth/image_raw"
CAMERA_INFO_TOPIC = f"/{CAMERA_NAMESPACE}/camera_info"

# 物体を中心に、この範囲でグリッパー(手先カメラの取り付け基準)をランダムにオフセットさせる。
# 大きすぎると物体が画角外に出て記録されるフレームが減り、小さすぎるとpicking_session.pyの
# 自然な軌道と変わらず偏りが解消されないため、手先カメラの実測画角を踏まえて決めている。
XY_JITTER_RANGE_M = 0.05
HEIGHT_MIN_M = 0.06
HEIGHT_MAX_M = 0.14
POSE_LOOKUP_TIMEOUT_SEC = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="data/wrist-camera/dataset/<version>のパス")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0, help="ファイル名の開始インデックス")
    parser.add_argument("--settle-sec", type=float, default=0.3, help="移動後、画像が安定するまでの待機秒数")
    return parser.parse_args()


class LatestFrameSubscriber:
    def __init__(self, node) -> None:
        self._bridge = CvBridge()
        self.camera_info: CameraInfo | None = None
        self.depth_image = None
        self.rgb_image = None
        node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 10)
        node.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)
        node.create_subscription(Image, RGB_TOPIC, self._on_rgb, 10)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_depth(self, msg: Image) -> None:
        self.depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_rgb(self, msg: Image) -> None:
        self.rgb_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def _lookup_pose(interface: RobotInterface, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        pose = interface.lookup_target_pose(TARGET_FRAME)
        if pose is not None:
            return pose
        time.sleep(0.1)
    return None


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("collect_wrist_jitter_dataset")
    interface = RobotInterface(node)
    frames = LatestFrameSubscriber(node)
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    writer = FrameWriter(args.output_dir)
    frame_count = 0

    try:
        while rclpy.ok() and frame_count < args.num_frames:
            object_pose = _lookup_pose(interface, POSE_LOOKUP_TIMEOUT_SEC)
            if object_pose is None:
                node.get_logger().error(f"{TARGET_FRAME}のTFが取得できず中断します")
                break

            dx, dy = np.random.uniform(-XY_JITTER_RANGE_M, XY_JITTER_RANGE_M, size=2)
            dz = np.random.uniform(HEIGHT_MIN_M, HEIGHT_MAX_M)
            target_xyz = (
                object_pose.position.x + dx,
                object_pose.position.y + dy,
                object_pose.position.z + dz,
            )
            orientation = downward_orientation_for_yaw(np.random.uniform(-math.pi, math.pi))

            try:
                interface.move_to(target_xyz, orientation)
            except RuntimeError as e:
                node.get_logger().info(f"移動をスキップ({e})")
                continue

            time.sleep(args.settle_sec)

            if frames.camera_info is None or frames.depth_image is None or frames.rgb_image is None:
                node.get_logger().info("カメラ画像が未受信のためスキップ")
                continue
            try:
                world_to_object = tf_buffer.lookup_transform(WORLD_FRAME, TARGET_FRAME, rclpy.time.Time())
                world_to_camera = tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_FRAME, rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue

            frame_name = f"{args.start_index + frame_count:04d}"
            wrote = writer.try_write_frame(
                frame_name,
                frames.rgb_image,
                frames.depth_image,
                frames.camera_info,
                world_to_object.transform,
                world_to_camera.transform,
            )
            if not wrote:
                continue

            frame_count += 1
            node.get_logger().info(f"frame {frame_name} recorded ({frame_count}/{args.num_frames})")
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=5.0)
        interface.moveit_py.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
