#!/usr/bin/env python3
"""共通のヘルパー"""

from collections.abc import Callable

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def safe_spin(node_factory: Callable[[], Node]) -> None:
    rclpy.init()
    node = node_factory()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, RCLError):
        # SIGTERM等による正常な終了経路。SIGTERMがspin_once()内部のwait_set生成と
        # 競合すると、ExternalShutdownExceptionではなく低レベルのRCLError
        # ("the given context is not valid")が飛んでくることがあるが、
        # どちらも外部からの終了要求による正常終了なので、ここで潰さないと
        # ros2runが異常終了扱いにする。
        pass
    finally:
        try:
            node.destroy_node()
        except RCLError:
            # 上と同じ競合でcontextが既に無効な場合、destroy_node()自体も失敗しうる。
            pass
        # 上のExternalShutdownExceptionは送出前に既にcontext.shutdown()が呼ばれているため、
        # ここでも無条件に呼ぶと「rcl_shutdown already called」で例外になる。
        if rclpy.ok():
            rclpy.shutdown()
