#!/usr/bin/env python3
"""camera/picking_session.py/wrist_camera/jitter.pyが共有するフレーム記録処理。

物体姿勢・カメラ姿勢からキーポイントラベルを生成しファイルへ書き出す部分は、
記録タイミングの決め方(トピックの流れに任せて記録するか、明示的に姿勢を
決めてから撮るか)に依存しないため、両スクリプトで共通化する。
"""

import json
import os

import cv2
import numpy as np
from common import target_object_shape as shape
from common.camera_intrinsics import intrinsics_from_camera_info
from common.transforms import transform_to_matrix
from tf_transformations import quaternion_matrix

# 学習データとして使うには、visible=2(遮蔽なし・画角内)の頂点が最低これだけ必要。
# 画角内に十分な頂点が写っていないフレーム(アームに完全遮蔽・画角外)は記録しない。
MIN_VISIBLE_CORNERS = 4


class FrameWriter:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._images_dir = os.path.join(output_dir, "images")
        self._labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._labels_dir, exist_ok=True)
        self._pose_gt_path = os.path.join(output_dir, "pose_gt.json")
        self._write_static_outputs()

    def try_write_frame(
        self, frame_name: str, rgb_image, depth_image, camera_info, object_transform, camera_transform
    ) -> bool:
        if not self._write_label(frame_name, depth_image, camera_info, object_transform, camera_transform):
            return False

        cv2.imwrite(os.path.join(self._images_dir, f"{frame_name}.png"), rgb_image)

        t = object_transform.translation
        r = object_transform.rotation
        self._append_pose_gt(
            {"frame": frame_name, "position_xyz": [t.x, t.y, t.z], "orientation_xyzw": [r.x, r.y, r.z, r.w]}
        )
        return True

    def _write_label(self, frame_name, depth_image, camera_info, object_transform, camera_transform) -> bool:
        fx, fy, cx, cy = intrinsics_from_camera_info(camera_info)
        width, height = camera_info.width, camera_info.height

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
            camera_points, fx, fy, cx, cy, width, height, depth_image, np
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
        keypoints_path = os.path.join(self._output_dir, "object_3d_keypoints.json")
        if not os.path.exists(keypoints_path):
            with open(keypoints_path, "w") as f:
                json.dump({"class": "target_object", "keypoints_local_xyz": shape.LOCAL_CORNERS}, f, indent=2)

        yaml_path = os.path.join(self._output_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            with open(yaml_path, "w") as f:
                f.write(shape.dataset_yaml_content(os.path.abspath(self._output_dir)))
