#!/usr/bin/env python3
"""YOLO11-Pose + PnPでtarget_objectの6D Poseを推定する(本番認識ノード)。

yolo_pose.pyが学習したモデルでRGB画像から物体の2Dキーポイントを検出し、
<MODEL_DIR>/object_3d_keypoints.json(物体ローカル座標での8頂点定義)と
組み合わせてcv2.solvePnP()で6D Poseを求める。モデル重み・キーポイント定義はどちらも
学習のたびに結果が変わりうるため、再現性のため気に入ったバージョンだけをdata/models/以下に
git管理下で保持している(それ以外の学習結果はdata/train-models/にローカルのみで残る)。
gt_tf_publisher_node/pose_estimation_node_classical_cvと全く同じworld -> target_objectの
TFを/tfへ配信することで、picking_controller_node側のコードを変更せずに認識方式を
差し替えられるようにする。
"""

import json
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from ultralytics import YOLO

from vision_picking.common import safe_spin

WORLD_FRAME = "world"
CAMERA_FRAME = "camera"
TARGET_FRAME = "target_object"
RGB_TOPIC = "/camera/rgb/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera_info"

# MODEL_VERSION環境変数(make vp-run-yolo VER=v1等)でどの学習結果を使うか切り替える。
# 未指定時はv1(data/models/、git管理下のpush済みモデル)を使う。"train:"で始まる場合は
# data/train-models/(ローカルのみ・push前の学習直後の結果)を指す。
_MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
if _MODEL_VERSION.startswith("train:"):
    _MODEL_DIR = f"/data/train-models/{_MODEL_VERSION[len('train:'):]}"
else:
    _MODEL_DIR = f"/data/models/{_MODEL_VERSION}"
MODEL_PATH = _MODEL_DIR + "/best.pt"
OBJECT_KEYPOINTS_PATH = _MODEL_DIR + "/object_3d_keypoints.json"

# 検出信頼度が低いキーポイントをPnPに混ぜると姿勢推定が不安定になるため足切りする。
# solvePnPには最低4点の対応が必要(8頂点は同一平面上にないため反転曖昧性は出にくいが、
# 4点未満では解自体が求まらない)。
MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_KEYPOINTS_FOR_PNP = 4
# キーポイント検出自体が破綻している(推定される画像上の位置と実際のキーポイントの対応が取れていない)場合、
# solvePnPは数値的には解を返すが物理的にありえない姿勢になることがある。
# 再投影誤差が大きい解は破棄する。
MAX_REPROJECTION_ERROR_PX = 15.0
# アームがカメラとtarget_objectの間を横切って大部分を遮蔽すると、モデルが見えない箇所の
# キーポイントを高い信頼度で誤って出力し、2D的には辻褄が合う(再投影誤差が小さい)まま
# 物理的にありえない深度に収束することがある。target_objectは常にテーブル面上
# (z≈0.025m)にあるため、worldフレームでのZがこの範囲を外れる解は破棄する。
EXPECTED_OBJECT_Z_M = 0.025
MAX_Z_ERROR_M = 0.05
# Zレンジチェックは明らかな暴走値を弾けるが、範囲内に収まる程度の不正確な値までは
# 検出できない(境界付近の値がそのまま把持位置のズレに直結し、把持失敗を招くことがある)。
# 直近STABILITY_WINDOW件が互いにSTABILITY_TOLERANCE_M以内に収まって初めて
# 「安定して検出できた」とみなし、TFを配信する。範囲チェックで弾かれたフレームがあれば
# 連続性が途切れたとみなし、蓄積をリセットする。
STABILITY_WINDOW = 3
STABILITY_TOLERANCE_M = 0.01


class PoseEstimationNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_estimation_node")
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._broadcaster = TransformBroadcaster(self)

        self._model = YOLO(MODEL_PATH)
        with open(OBJECT_KEYPOINTS_PATH) as f:
            keypoints_data = json.load(f)
        self._object_points = np.array(keypoints_data["keypoints_local_xyz"], dtype=np.float64)
        self._recent_translations: list[np.ndarray] = []

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 10)
        self.create_subscription(Image, RGB_TOPIC, self._on_rgb, 10)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_rgb(self, msg: Image) -> None:
        if self._camera_info is None:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self._model.predict(frame, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return

        # 複数検出された場合はbox信頼度が最も高い1件を採用する。
        best_idx = int(result.boxes.conf.argmax())
        image_points = result.keypoints.xy[best_idx].cpu().numpy()
        if result.keypoints.conf is not None:
            kpt_confidences = result.keypoints.conf[best_idx].cpu().numpy()
        else:
            kpt_confidences = np.ones(len(image_points))

        valid = kpt_confidences >= MIN_KEYPOINT_CONFIDENCE
        if np.count_nonzero(valid) < MIN_KEYPOINTS_FOR_PNP:
            self.get_logger().info("検出キーポイント数が不足しているためTF配信をスキップ")
            return

        camera_matrix = np.array(self._camera_info.k, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.array(self._camera_info.d, dtype=np.float64) if self._camera_info.d else np.zeros(5)

        ok, rvec, tvec = cv2.solvePnP(
            self._object_points[valid], image_points[valid], camera_matrix, dist_coeffs
        )
        if not ok:
            return

        reprojected, _ = cv2.projectPoints(self._object_points[valid], rvec, tvec, camera_matrix, dist_coeffs)
        reprojection_error = float(np.linalg.norm(reprojected.reshape(-1, 2) - image_points[valid], axis=1).mean())
        if reprojection_error > MAX_REPROJECTION_ERROR_PX:
            self.get_logger().info(
                f"再投影誤差が大きい({reprojection_error:.1f}px)ためTF配信をスキップ"
            )
            return

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # solvePnPが返すのはtarget_object -> cameraの変換(物体座標系での点をカメラ座標系へ写す行列)なので、
        # この後world -> cameraと合成してworld -> target_objectへ変換する。
        object_to_camera = np.eye(4)
        object_to_camera[:3, :3] = rotation_matrix
        object_to_camera[:3, 3] = tvec.flatten()

        self._publish_in_world_frame(object_to_camera)

    def _publish_in_world_frame(self, object_to_camera: np.ndarray) -> None:
        try:
            world_to_camera = self._tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().info(f"{CAMERA_FRAME}のTFを待機中...")
            return

        world_to_object_mat = _transform_to_matrix(world_to_camera.transform) @ object_to_camera
        translation = world_to_object_mat[:3, 3]
        if abs(float(translation[2]) - EXPECTED_OBJECT_Z_M) > MAX_Z_ERROR_M:
            self.get_logger().info(
                f"Zが物理的にありえない範囲({translation[2]:.3f}m)のためTF配信をスキップ"
            )
            self._recent_translations.clear()
            return

        self._recent_translations.append(translation.copy())
        if len(self._recent_translations) > STABILITY_WINDOW:
            self._recent_translations.pop(0)
        if len(self._recent_translations) < STABILITY_WINDOW:
            return
        spread = max(
            float(np.linalg.norm(a - b))
            for i, a in enumerate(self._recent_translations)
            for b in self._recent_translations[i + 1 :]
        )
        if spread > STABILITY_TOLERANCE_M:
            self.get_logger().info(
                f"検出が安定していない(直近{STABILITY_WINDOW}件のばらつき{spread:.3f}m)ためTF配信をスキップ"
            )
            return

        qx, qy, qz, qw = quaternion_from_matrix(world_to_object_mat)

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = WORLD_FRAME
        out.child_frame_id = TARGET_FRAME
        out.transform.translation.x = float(translation[0])
        out.transform.translation.y = float(translation[1])
        out.transform.translation.z = float(translation[2])
        out.transform.rotation.x = qx
        out.transform.rotation.y = qy
        out.transform.rotation.z = qz
        out.transform.rotation.w = qw
        self._broadcaster.sendTransform(out)


def _transform_to_matrix(transform) -> np.ndarray:
    mat = quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    mat[0, 3] = transform.translation.x
    mat[1, 3] = transform.translation.y
    mat[2, 3] = transform.translation.z
    return mat


def main() -> None:
    safe_spin(PoseEstimationNode)


if __name__ == "__main__":
    main()
