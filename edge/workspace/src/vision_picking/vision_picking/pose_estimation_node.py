#!/usr/bin/env python3
"""YOLO11-Pose + PnPでtarget_objectの6D Poseを推定する(本番認識ノード)。

yolo_pose.pyが学習したモデルでRGB画像から物体の2Dキーポイントを検出し、
<MODEL_DIR>/object_3d_keypoints.json(物体ローカル座標での8頂点定義)と
組み合わせてcv2.solvePnP()で6D Poseを求める。モデル重み・キーポイント定義はどちらも
学習のたびに結果が変わりうるため、再現性のため気に入ったバージョンだけをdata/<camera>/models/以下に
git管理下で保持している(それ以外の学習結果はdata/<camera>/train-models/にローカルのみで残る)。
gt_tf_publisher_node/pose_estimation_node_classical_cvと全く同じworld -> target_objectの
TFを/tfへ配信することで、picking_controller_node側のコードを変更せずに認識方式を
差し替えられるようにする。

CAMERA_NAMESPACE環境変数で、俯瞰カメラ(既定値"camera")・手先カメラ("wrist_camera"等)の
どちらでも同じスクリプトを使い回せる(camera_bridge_node.pyが中継するトピック名・frame_idの
命名規則に合わせている)。粗検出(俯瞰)・精緻検出(手先)を同時に動かして両方の結果を
区別したい場合は、TARGET_FRAME_OUT環境変数で出力先のTF子フレーム名も分ける
(例: target_object_coarse / target_object_fine)。
"""

import json
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import Color, ImageAnnotations, Point2, PointsAnnotation
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_matrix
from ultralytics import YOLO

from common.target_object_shape import OBJECT_SIZE_M, TABLE_HEIGHT_M
from common.transforms import transform_to_matrix
from vision_picking.common import safe_spin

WORLD_FRAME = "world"
# camera_bridge_node.pyの中継先(/{namespace}/rgb/image_raw等、frame_id={namespace})と
# 命名規則を揃えている。
CAMERA_NAMESPACE = os.environ.get("CAMERA_NAMESPACE", "camera")
CAMERA_FRAME = CAMERA_NAMESPACE
RGB_TOPIC = f"/{CAMERA_NAMESPACE}/rgb/image_raw"
CAMERA_INFO_TOPIC = f"/{CAMERA_NAMESPACE}/camera_info"
# Foxglove StudioのImageパネルで/rgb/image_rawに重ねて表示できる2D注釈(検出キーポイント・
# PnP再投影点)を配信する。デバッグ用で認識パイプライン自体はこのトピックを購読しない。
ANNOTATIONS_TOPIC = f"/{CAMERA_NAMESPACE}/rgb/annotations"
DETECTED_KEYPOINT_COLOR = Color(r=0.2, g=1.0, b=0.2, a=1.0)
LOW_CONFIDENCE_KEYPOINT_COLOR = Color(r=1.0, g=0.2, b=0.2, a=0.8)
REPROJECTED_POINT_COLOR = Color(r=0.2, g=0.6, b=1.0, a=1.0)
# 俯瞰(粗検出)・手先(精緻検出)を同時に動かす場合に出力先を区別できるようにする。
TARGET_FRAME = os.environ.get("TARGET_FRAME_OUT", "target_object")
# 同時に2インスタンス動かす場合にROS2ノード名が衝突しないよう、既定(俯瞰カメラ)以外は
# サフィックスを付ける。俯瞰カメラ単独運用時の既存の挙動(ノード名)は変えない。
NODE_NAME = "pose_estimation_node" if CAMERA_NAMESPACE == "camera" else f"pose_estimation_node_{CAMERA_NAMESPACE}"

# data/以下はカメラ(overhead-camera/wrist-camera)ごとに分かれているため、
# CAMERA_NAMESPACEから対応するディレクトリ名を導出する。
_MODEL_CAMERA_DIR = "overhead-camera" if CAMERA_NAMESPACE == "camera" else "wrist-camera"

# MODEL_VERSION環境変数(make vp-run-yolo VER_COARSE=v1等)でどの学習結果を使うか切り替える。
# 未指定時はv1(data/<camera>/models/、git管理下のpush済みモデル)を使う。"train:"で始まる場合は
# data/<camera>/train-models/(ローカルのみ・push前の学習直後の結果)を指す。
_MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
if _MODEL_VERSION.startswith("train:"):
    _MODEL_DIR = f"/data/{_MODEL_CAMERA_DIR}/train-models/{_MODEL_VERSION[len('train:'):]}"
else:
    _MODEL_DIR = f"/data/{_MODEL_CAMERA_DIR}/models/{_MODEL_VERSION}"
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
# Ref: https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
# solvePnP単体だと、一部のキーポイントの検出ノイズにPnP解全体が引っ張られ、
# 再投影誤差の平均としては閾値内に収まっていても特定方向(特にZ)へ系統的にバイアスした
# 姿勢に収束することがある(近距離で見る手先カメラほど影響を受けやすい)。
# solvePnPRansacで外れ値になりやすい点を先に切り離してから解く。
RANSAC_INLIER_THRESHOLD_PX = 8.0
# アームがカメラとtarget_objectの間を横切って大部分を遮蔽すると、モデルが見えない箇所の
# キーポイントを高い信頼度で誤って出力し、2D的には辻褄が合う(再投影誤差が小さい)まま
# 物理的にありえない深度に収束することがある。target_objectは常に作業台面上に
# あるため、worldフレームでのZがこの範囲を外れる解は破棄する。
EXPECTED_OBJECT_Z_M = TABLE_HEIGHT_M + OBJECT_SIZE_M / 2.0
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
        super().__init__(NODE_NAME)
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
        self._recent_rotations: list[np.ndarray] = []

        self._annotations_pub = self.create_publisher(ImageAnnotations, ANNOTATIONS_TOPIC, 10)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, 10)
        self.create_subscription(Image, RGB_TOPIC, self._on_rgb, 10)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    @staticmethod
    def _points_annotation(stamp, points_xy: np.ndarray, colors: list[Color]) -> PointsAnnotation:
        annotation = PointsAnnotation()
        annotation.timestamp = stamp
        annotation.type = PointsAnnotation.POINTS
        annotation.points = [Point2(x=float(x), y=float(y)) for x, y in points_xy]
        annotation.outline_colors = colors
        annotation.thickness = 6.0
        return annotation

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
        keypoint_colors = [
            DETECTED_KEYPOINT_COLOR if is_valid else LOW_CONFIDENCE_KEYPOINT_COLOR for is_valid in valid
        ]
        self._annotations_pub.publish(
            ImageAnnotations(points=[self._points_annotation(msg.header.stamp, image_points, keypoint_colors)])
        )

        if np.count_nonzero(valid) < MIN_KEYPOINTS_FOR_PNP:
            self.get_logger().info("検出キーポイント数が不足しているためTF配信をスキップ")
            return

        camera_matrix = np.array(self._camera_info.k, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.array(self._camera_info.d, dtype=np.float64) if self._camera_info.d else np.zeros(5)

        try:
            # solvePnPRansac()もsolvePnP()同様、内部のDLT初期化が有効点数によっては
            # (4点以上でもMIN_KEYPOINTS_FOR_PNPを満たしていても)cv2.errorを送出することが
            # あるため、検出不良の1フレームとして扱いスキップする。
            ok, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
                self._object_points[valid],
                image_points[valid],
                camera_matrix,
                dist_coeffs,
                reprojectionError=RANSAC_INLIER_THRESHOLD_PX,
            )
        except cv2.error:
            return
        if not ok or inlier_indices is None or len(inlier_indices) < MIN_KEYPOINTS_FOR_PNP:
            return
        inlier_indices = inlier_indices.flatten()
        inlier_object_points = self._object_points[valid][inlier_indices]
        inlier_image_points = image_points[valid][inlier_indices]

        reprojected, _ = cv2.projectPoints(inlier_object_points, rvec, tvec, camera_matrix, dist_coeffs)
        reprojection_error = float(np.linalg.norm(reprojected.reshape(-1, 2) - inlier_image_points, axis=1).mean())
        self._annotations_pub.publish(
            ImageAnnotations(
                points=[
                    self._points_annotation(msg.header.stamp, image_points, keypoint_colors),
                    self._points_annotation(
                        msg.header.stamp,
                        reprojected.reshape(-1, 2),
                        [REPROJECTED_POINT_COLOR] * len(reprojected),
                    ),
                ]
            )
        )
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

        world_to_object_mat = transform_to_matrix(world_to_camera.transform) @ object_to_camera
        translation = world_to_object_mat[:3, 3]
        if abs(float(translation[2]) - EXPECTED_OBJECT_Z_M) > MAX_Z_ERROR_M:
            self.get_logger().info(
                f"Zが物理的にありえない範囲({translation[2]:.3f}m)のためTF配信をスキップ"
            )
            self._recent_translations.clear()
            self._recent_rotations.clear()
            return

        self._recent_translations.append(translation.copy())
        self._recent_rotations.append(world_to_object_mat[:3, :3].copy())
        if len(self._recent_translations) > STABILITY_WINDOW:
            self._recent_translations.pop(0)
            self._recent_rotations.pop(0)
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

        # 収束判定に使った直近STABILITY_WINDOW件は、単発の検出ノイズを含んだまま
        # ばらつきの許容範囲内に収まっているだけなので、その平均を配信することで
        # 単一フレームの検出ノイズを均してから使う(下流のグリップ位置精度・向き精度に直結する)。
        averaged_translation = np.mean(self._recent_translations, axis=0)
        # 回転行列は単純平均すると直交行列でなくなる(スケールが崩れる)ため、
        # 平均後にSVDで最も近い回転行列へ再直交化する(回転averagingの標準的な手法)。
        u, _, vt = np.linalg.svd(np.mean(self._recent_rotations, axis=0))
        averaged_rotation = u @ vt
        if np.linalg.det(averaged_rotation) < 0:
            u[:, -1] *= -1
            averaged_rotation = u @ vt
        averaged_mat = np.eye(4)
        averaged_mat[:3, :3] = averaged_rotation
        qx, qy, qz, qw = quaternion_from_matrix(averaged_mat)

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = WORLD_FRAME
        out.child_frame_id = TARGET_FRAME
        out.transform.translation.x = float(averaged_translation[0])
        out.transform.translation.y = float(averaged_translation[1])
        out.transform.translation.z = float(averaged_translation[2])
        out.transform.rotation.x = qx
        out.transform.rotation.y = qy
        out.transform.rotation.z = qz
        out.transform.rotation.w = qw
        self._broadcaster.sendTransform(out)


def main() -> None:
    safe_spin(PoseEstimationNode)


if __name__ == "__main__":
    main()
