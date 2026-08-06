#!/usr/bin/env python3
"""geometry_msgs/Transformまわりの共通処理。

vision_pickingパッケージのROS2ノード(pose_estimation_node.py等)と、
scripts/dataset/配下のcolcon管理外スクリプト(camera/picking_session.py)の両方が
全く同じworld<->camera<->target_objectの変換行列計算を必要とするため、
どちらか一方の下に置くと重複・食い違いが起きやすい。src直下の独立したパッケージに置き、
両方から`from common.transforms import transform_to_matrix`で参照する。
"""

import numpy as np
from tf_transformations import quaternion_matrix


def transform_to_matrix(transform) -> np.ndarray:
    """geometry_msgs/Transformを4x4同次変換行列に変換する。"""
    mat = quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
    mat[0, 3] = transform.translation.x
    mat[1, 3] = transform.translation.y
    mat[2, 3] = transform.translation.z
    return mat
