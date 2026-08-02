#!/usr/bin/env python3
"""共通のヘルパー"""

from collections.abc import Callable

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def safe_spin(node_factory: Callable[[], Node]) -> None:
    rclpy.init()
    node = node_factory()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        # SIGTERM等による正常な終了経路。ここで潰さないとros2runが異常終了扱いにする。
        pass
    finally:
        node.destroy_node()
        # 上のExternalShutdownExceptionは送出前に既にcontext.shutdown()が呼ばれているため、
        # ここでも無条件に呼ぶと「rcl_shutdown already called」で例外になる。
        if rclpy.ok():
            rclpy.shutdown()
