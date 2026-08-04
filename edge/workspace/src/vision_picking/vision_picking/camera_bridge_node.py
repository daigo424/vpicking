#!/usr/bin/env python3
"""run_simulation.pyのOmniGraphが配信するカメラデータを認識ノード向けの安定したトピック名へ中継する。

run_simulation.py側のROS2 Camera Helperは実装依存のトピック名でpublishしている。
実カメラのドライバに差し替える場合でも認識ノード側の購読先が変わらずに済むよう、
安定したトピック名へこのノードが中継する。俯瞰カメラ・手先カメラの両方を扱う。

手先カメラはアームの動きに追従して常にworld系での姿勢が変化するため、
/ground_truth/tf(gt_tf_publisher_nodeと同じソース)を受け取るたびに
world -> <camera frame>のTFを継続的に再配信する(俯瞰カメラは実際には静止しているが、
同じ経路で継続配信しても実害はないため区別しない)。
"""

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from vision_picking.common import safe_spin

GROUND_TRUTH_TOPIC = "/ground_truth/tf"
PARENT_FRAME = "world"

# run_simulation.pyのbuild_action_graph()に渡すカメラ設定(topic_prefix/frame_id)と対応させる。
CAMERAS = [
    {"src_prefix": "/sim_camera", "dst_prefix": "/camera", "frame_id": "camera"},
    {"src_prefix": "/sim_wrist_camera", "dst_prefix": "/wrist_camera", "frame_id": "wrist_camera"},
]


class CameraBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_bridge_node")
        self._broadcaster = TransformBroadcaster(self)
        self._frame_ids = {camera["frame_id"] for camera in CAMERAS}

        for camera in CAMERAS:
            rgb_pub = self.create_publisher(Image, f"{camera['dst_prefix']}/rgb/image_raw", 10)
            depth_pub = self.create_publisher(Image, f"{camera['dst_prefix']}/depth/image_raw", 10)
            camera_info_pub = self.create_publisher(CameraInfo, f"{camera['dst_prefix']}/camera_info", 10)
            self.create_subscription(Image, f"{camera['src_prefix']}/rgb", rgb_pub.publish, 10)
            self.create_subscription(Image, f"{camera['src_prefix']}/depth", depth_pub.publish, 10)
            self.create_subscription(CameraInfo, f"{camera['src_prefix']}/camera_info", camera_info_pub.publish, 10)

        self.create_subscription(TFMessage, GROUND_TRUTH_TOPIC, self._on_ground_truth_tf, 10)

    def _on_ground_truth_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id not in self._frame_ids:
                continue
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = PARENT_FRAME
            out.child_frame_id = transform.child_frame_id
            out.transform = transform.transform
            self._broadcaster.sendTransform(out)


def main() -> None:
    safe_spin(CameraBridgeNode)


if __name__ == "__main__":
    main()
