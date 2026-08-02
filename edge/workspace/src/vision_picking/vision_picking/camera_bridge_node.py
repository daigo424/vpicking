#!/usr/bin/env python3
"""run_simulation.pyのOmniGraphが配信するカメラデータを認識ノード向けの安定したトピック名へ中継する。

run_simulation.py側のROS2 Camera Helperは/sim_camera/*という実装依存のトピック名で
publishしている。実カメラのドライバに差し替える場合でも認識ノード側の購読先が変わらずに
済むよう、/camera/*という安定したトピック名へこのノードが中継する。

カメラのworld系での姿勢は静止しているため、/ground_truth/tf(gt_tf_publisher_nodeと同じ
ソース)から最初の1回だけ拾ってworld -> cameraの静的TFとして/tf_staticに配信する。
"""

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster

from vision_picking.common import safe_spin

GROUND_TRUTH_TOPIC = "/ground_truth/tf"
PARENT_FRAME = "world"
CAMERA_FRAME = "camera"

RGB_SRC_TOPIC = "/sim_camera/rgb"
RGB_DST_TOPIC = "/camera/rgb/image_raw"
DEPTH_SRC_TOPIC = "/sim_camera/depth"
DEPTH_DST_TOPIC = "/camera/depth/image_raw"
CAMERA_INFO_SRC_TOPIC = "/sim_camera/camera_info"
CAMERA_INFO_DST_TOPIC = "/camera/camera_info"


class CameraBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_bridge_node")
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._camera_tf_published = False

        self._rgb_pub = self.create_publisher(Image, RGB_DST_TOPIC, 10)
        self._depth_pub = self.create_publisher(Image, DEPTH_DST_TOPIC, 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, CAMERA_INFO_DST_TOPIC, 10)

        self.create_subscription(Image, RGB_SRC_TOPIC, self._rgb_pub.publish, 10)
        self.create_subscription(Image, DEPTH_SRC_TOPIC, self._depth_pub.publish, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_SRC_TOPIC, self._camera_info_pub.publish, 10)
        self.create_subscription(TFMessage, GROUND_TRUTH_TOPIC, self._on_ground_truth_tf, 10)

    def _on_ground_truth_tf(self, msg: TFMessage) -> None:
        if self._camera_tf_published:
            return
        for transform in msg.transforms:
            if CAMERA_FRAME not in transform.child_frame_id:
                continue
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = PARENT_FRAME
            out.child_frame_id = CAMERA_FRAME
            out.transform = transform.transform
            self._static_broadcaster.sendTransform(out)
            self._camera_tf_published = True
            return


def main() -> None:
    safe_spin(CameraBridgeNode)


if __name__ == "__main__":
    main()
