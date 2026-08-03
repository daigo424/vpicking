#!/usr/bin/env python3
"""深度点群のセグメンテーション + 重心・主軸(PCA)推定でtarget_objectの6D Poseを推定する。

gt_tf_publisher_nodeと全く同じworld -> target_objectのTFを/tfへ配信することで、
picking_controller_node側のコードを変更せずに認識方式を差し替えられるようにする。
学習済みモデルの代わりに、テーブル面との深度差でtarget_objectを分離する軽量な
古典的CV手法を使う。
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_euler, quaternion_from_matrix

from common.camera_intrinsics import intrinsics_from_camera_info
from common.target_object_shape import OBJECT_SIZE_M as TARGET_OBJECT_SIZE_M
from common.transforms import transform_to_matrix
from vision_picking.common import safe_spin

WORLD_FRAME = "world"
CAMERA_FRAME = "camera"
TARGET_FRAME = "target_object"
DEPTH_TOPIC = "/camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera_info"

# target_objectは5cm角のcubeで、カメラは真上から見下ろしている。
# テーブル面はカメラの深度画像の大部分を占めるため、中央値をテーブル面の深度とみなす。
# カメラの視野にはPandaアーム自体も入り込むため、
# 「テーブルより近ければ物体」という単純な閾値だとテーブルより遥かに手前にあるアームを物体として誤検出する。
# 物体の高さ(5cm角)を上回らない範囲に上限も設けて、テーブル直上の薄い depth の層だけを物体とみなす。
DEPTH_MARGIN_MIN_M = 0.02
DEPTH_MARGIN_MAX_M = 0.10
MIN_OBJECT_PIXELS = 30


class PoseEstimationClassicalCvNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_estimation_node_classical_cv")
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._broadcaster = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_depth(self, msg: Image) -> None:
        if self._camera_info is None:
            return

        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        fx, fy, cx, cy = intrinsics_from_camera_info(self._camera_info)

        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return
        table_depth = float(np.median(depth[valid]))
        object_mask = (
            valid
            & (depth < table_depth - DEPTH_MARGIN_MIN_M)
            & (depth > table_depth - DEPTH_MARGIN_MAX_M)
        )
        if np.count_nonzero(object_mask) < MIN_OBJECT_PIXELS:
            return

        vs, us = np.nonzero(object_mask)
        zs = depth[vs, us]
        # Ref: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/reference_material/reference_conventions.html
        # 「ROS 2へpublishされるカメラ関連のデータ(姿勢のTF含む)はROS軸(X右, Y下, Z前方)で表現される」と
        # Isaac Simのドキュメントが明記しているため、
        # cameraフレームに対して通常のpinholeモデルの変換式(X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=depth)をそのまま使える。
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        points_camera = np.stack([xs, ys, zs], axis=1)
        centroid_camera = points_camera.mean(axis=0)

        # 主軸(PCA): target_objectは正方形のcubeで主軸自体に強い意味はないが、
        # 「深度点群のセグメンテーション+重心+主軸推定」という認識方式そのものを実証するために計算する。
        centered_xy = points_camera[:, :2] - centroid_camera[:2]
        eigvals, eigvecs = np.linalg.eigh(np.cov(centered_xy.T))
        principal_axis = eigvecs[:, np.argmax(eigvals)]
        yaw = float(np.arctan2(principal_axis[1], principal_axis[0]))

        pose_camera = TransformStamped()
        pose_camera.header.frame_id = CAMERA_FRAME
        pose_camera.child_frame_id = TARGET_FRAME
        pose_camera.transform.translation.x = float(centroid_camera[0])
        pose_camera.transform.translation.y = float(centroid_camera[1])
        pose_camera.transform.translation.z = float(centroid_camera[2])
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        pose_camera.transform.rotation.x = qx
        pose_camera.transform.rotation.y = qy
        pose_camera.transform.rotation.z = qz
        pose_camera.transform.rotation.w = qw

        self._publish_in_world_frame(pose_camera)

    def _publish_in_world_frame(self, pose_camera: TransformStamped) -> None:
        try:
            world_to_camera = self._tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().info(f"{CAMERA_FRAME}のTFを待機中...")
            return

        world_to_object_mat = transform_to_matrix(world_to_camera.transform) @ transform_to_matrix(
            pose_camera.transform
        )
        translation = world_to_object_mat[:3, 3]
        qx, qy, qz, qw = quaternion_from_matrix(world_to_object_mat)

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = WORLD_FRAME
        out.child_frame_id = TARGET_FRAME
        out.transform.translation.x = float(translation[0])
        out.transform.translation.y = float(translation[1])
        # 深度カメラは物体の上面しか観測できないが、picking_controller_nodeはtarget_objectの
        # フレーム原点を物体の重心として掴みに行く高さを計算している。上面の重心をそのまま
        # 使うと指が物体の上端しか捉えられず把持に失敗するため、物体サイズの半分だけ
        # 下げて重心相当の高さに補正する。
        out.transform.translation.z = float(translation[2]) - TARGET_OBJECT_SIZE_M / 2.0
        out.transform.rotation.x = qx
        out.transform.rotation.y = qy
        out.transform.rotation.z = qz
        out.transform.rotation.w = qw
        self._broadcaster.sendTransform(out)


def main() -> None:
    safe_spin(PoseEstimationClassicalCvNode)


if __name__ == "__main__":
    main()
