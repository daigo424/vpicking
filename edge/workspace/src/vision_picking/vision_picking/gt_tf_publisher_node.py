#!/usr/bin/env python3
"""run_simulation.pyのOmniGraphが配信するtarget_objectの真値TFをworld -> target_objectとして
再配信するground truthノード。

run_simulation.py側のROS2PublishTransformTreeは/ground_truth/tf(このノード専用の入力)に
publishしており、最終的な/tfには直接publishしない。認識方式を変える場合でも
world -> target_objectを/tfに配信するノードを差し替えるだけで済むようにするため。
"""

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from vision_picking.common import safe_spin

GROUND_TRUTH_TOPIC = "/ground_truth/tf"
PARENT_FRAME = "world"
CHILD_FRAME = "target_object"


class GtTfPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("gt_tf_publisher_node")
        self._broadcaster = TransformBroadcaster(self)
        # Ref: https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_putting_it_all_together.html#building-omnigraph-nodes
        self.create_subscription(TFMessage, GROUND_TRUTH_TOPIC, self._on_ground_truth_tf, 10)

    def _on_ground_truth_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if CHILD_FRAME not in transform.child_frame_id:
                continue
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = PARENT_FRAME
            out.child_frame_id = CHILD_FRAME
            out.transform = transform.transform
            self._broadcaster.sendTransform(out) # /tfにpublish


def main() -> None:
    safe_spin(GtTfPublisherNode)


if __name__ == "__main__":
    main()
