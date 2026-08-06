#!/usr/bin/env python3
"""Foxglove StudioのDiagnosticsパネルが標準で購読する"/diagnostics"へ、各コントローラノードの
実行状況をDiagnosticArrayとして配信するための共通ヘルパー。"""

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

DIAGNOSTICS_TOPIC = "/diagnostics"


def build_diagnostics(node, hardware_id: str, statuses: list[tuple[str, int, str]]) -> DiagnosticArray:
    msg = DiagnosticArray()
    msg.header.stamp = node.get_clock().now().to_msg()
    for name, level, message in statuses:
        status = DiagnosticStatus()
        status.hardware_id = hardware_id
        status.name = name
        status.level = level
        status.message = message
        msg.status.append(status)
    return msg
