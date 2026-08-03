#!/usr/bin/env python3
"""sensor_msgs/CameraInfoまわりの共通処理。"""


def intrinsics_from_camera_info(camera_info) -> tuple[float, float, float, float]:
    """sensor_msgs/CameraInfoのK(3x3行優先: [fx,0,cx, 0,fy,cy, 0,0,1])からfx, fy, cx, cyを取り出す。"""
    fx, fy = camera_info.k[0], camera_info.k[4]
    cx, cy = camera_info.k[2], camera_info.k[5]
    return fx, fy, cx, cy
