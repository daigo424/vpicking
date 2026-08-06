#!/usr/bin/env python3
"""edge_containerを操作してYOLO11-Pose学習用データセットを収集する、ホスト側スクリプト。

以前はcollect_overhead_picking_session_dataset.sh/collect_wrist_picking_session_dataset.sh/
collect_wrist_jitter_dataset.sh/collect_overhead_cube_only_dataset.shの4本の
ほぼ同じ内容のシェルスクリプトに分かれていたが、共通のsim起動〜ready待ち〜
補助ノード起動〜クリーンアップの流れを1つにまとめ、収集方式(--method)で分岐する。

data/はdocker-compose.local.ymlで../../data:/dataとバインドマウントされているため、
ホスト上のこのスクリプトから直接次バージョンを採番できる。
"""

import argparse
import math
import os
import random
import re
import subprocess
import time

CONTAINER = "edge_container"
SETUP = "source /workspace/install/setup.bash 2>/dev/null"

# カメラは(0.5, 0.0, 0.8)固定・真下向きで、実測画角(fx=fy=1527px, キューブ上面までの
# 距離0.75m)は約31cm x 24cm。yaw回転時のキューブ最大到達距離(対角の半分、約3.5cm)と
# 1cmの安全マージンを差し引いた、中心が確実に画角内に収まる範囲が以下。
# これより広い範囲でランダム配置すると、画角外でフレームが1枚も撮れないイテレーションが多発する。
# picking-session/jitterどちらも同じ値を流用している(jitter側は手先カメラの相対位置で
# 見えるためこの範囲である必要はないが、俯瞰カメラ向けで実績のある範囲をそのまま使う)。
SPAWN_X_RANGE = (0.388, 0.612)
SPAWN_Y_RANGE = (-0.073, 0.073)


def docker_exec(cmd: str, detach: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess:
    args = ["docker", "exec"] + (["-d"] if detach else []) + [CONTAINER, "bash", "-c", cmd]
    return subprocess.run(args, capture_output=not detach, text=True, timeout=timeout)


def next_version_dir(host_dir: str) -> str:
    if not os.path.isdir(host_dir):
        return "v1"
    numbers = [
        int(m.group(1))
        for name in os.listdir(host_dir)
        if (m := re.fullmatch(r"v(\d+)", name)) and os.path.isdir(os.path.join(host_dir, name))
    ]
    return f"v{max(numbers, default=0) + 1}"


def cleanup(extra_patterns: tuple[str, ...] = ()) -> None:
    patterns = (
        "run_simulation.py",
        "vision_picking/lib/vision_picking/gt_tf_publisher_node",
        "vision_picking/lib/vision_picking/camera_bridge_node",
        "vision_picking/lib/vision_picking/picking_controller_node",
        *extra_patterns,
    )
    for pattern in patterns:
        docker_exec(f'pkill -9 -f "{pattern}"')
    time.sleep(1)


def wait_for_ground_truth(timeout_sec: float = 120.0) -> bool:
    # ros2 topic hzは自分から終了しないため、timeout Nを付けても常にフルN秒待ってから
    # 強制終了される。準備できればすぐ1件返るtopic echo --onceの方がポーリングに向く。
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = docker_exec(f"{SETUP}; timeout 5 ros2 topic echo /ground_truth/tf --once 2>/dev/null")
        if "target_object" in (result.stdout or ""):
            return True
        time.sleep(2)
    return False


def launch_sim_and_wait(target_x: float, target_y: float, target_yaw: float) -> bool:
    docker_exec(
        f"export OMNI_KIT_ACCEPT_EULA=YES TARGET_OBJECT_X={target_x} TARGET_OBJECT_Y={target_y} "
        f"TARGET_OBJECT_YAW={target_yaw}; {SETUP}; cd /workspace && pixi run sim --headless",
        detach=True,
    )
    return wait_for_ground_truth()


def image_count(output_dir: str) -> str:
    result = docker_exec(f"ls {output_dir}/images 2>/dev/null | wc -l")
    return (result.stdout or "").strip()


def collect_picking_session(camera: str, iterations: int, frames_per_run: int) -> None:
    camera_dir = "overhead-camera" if camera == "overhead-camera" else "wrist-camera"
    version = next_version_dir(f"data/{camera_dir}/dataset/picking_session")
    output_dir = f"/data/{camera_dir}/dataset/picking_session/{version}"
    print(f"出力先: {output_dir}")

    camera_env = "export CAMERA_NAMESPACE=wrist_camera; " if camera == "wrist-camera" else ""

    for i in range(iterations):
        cleanup(extra_patterns=("picking_session.py",))
        x = round(random.uniform(*SPAWN_X_RANGE), 3)
        y = round(random.uniform(*SPAWN_Y_RANGE), 3)
        yaw = round(random.uniform(-math.pi, math.pi), 3)
        start_index = i * frames_per_run
        print(f"=== iteration {i}/{iterations - 1}: x={x} y={y} yaw={yaw} start_index={start_index} ===")

        if not launch_sim_and_wait(x, y, yaw):
            print("  sim起動タイムアウト、スキップ")
            continue

        docker_exec(f"{SETUP}; cd /workspace && pixi run gt_tf_publisher_node", detach=True)
        docker_exec(f"{SETUP}; cd /workspace && pixi run camera_bridge_node", detach=True)
        time.sleep(3)
        docker_exec(
            f"{camera_env}{SETUP}; cd /workspace && pixi run picking_session "
            f"--output-dir {output_dir} --max-frames {frames_per_run} "
            f"--start-index {start_index} --min-interval-sec 0.15",
            detach=True,
        )
        time.sleep(1)

        result = docker_exec(f"{SETUP}; cd /workspace && timeout 30 pixi run picking_controller_node")
        print("\n".join((result.stdout or "").splitlines()[-3:]))

        time.sleep(2)
        print(f"  現在の総フレーム数: {image_count(output_dir)}")

    cleanup(extra_patterns=("picking_session.py",))
    print("=== 収集完了 ===")
    print(image_count(output_dir))


def collect_jitter(iterations: int, frames_per_run: int) -> None:
    version = next_version_dir("data/wrist-camera/dataset/jitter")
    output_dir = f"/data/wrist-camera/dataset/jitter/{version}"
    print(f"出力先: {output_dir}")

    for i in range(iterations):
        cleanup(extra_patterns=("jitter.py",))
        x = round(random.uniform(*SPAWN_X_RANGE), 3)
        y = round(random.uniform(*SPAWN_Y_RANGE), 3)
        yaw = round(random.uniform(-math.pi, math.pi), 3)
        start_index = i * frames_per_run
        print(f"=== iteration {i}/{iterations - 1}: x={x} y={y} yaw={yaw} start_index={start_index} ===")

        if not launch_sim_and_wait(x, y, yaw):
            print("  sim起動タイムアウト、スキップ")
            continue

        docker_exec(f"{SETUP}; cd /workspace && pixi run gt_tf_publisher_node", detach=True)
        docker_exec(f"{SETUP}; cd /workspace && pixi run camera_bridge_node", detach=True)
        time.sleep(3)

        result = docker_exec(
            f"{SETUP}; cd /workspace && timeout 120 pixi run jitter "
            f"--output-dir {output_dir} --num-frames {frames_per_run} --start-index {start_index}"
        )
        print("\n".join((result.stdout or "").splitlines()[-10:]))

        print(f"  現在の総フレーム数: {image_count(output_dir)}")

    cleanup(extra_patterns=("jitter.py",))
    print("=== 収集完了 ===")
    print(image_count(output_dir))


def collect_cube_only(num_frames: int) -> None:
    # cube_only.pyは自前でSimulationAppを起動する自己完結型のスクリプトのため、
    # 他の収集方式と違いsim起動待ち・gt_tf_publisher_node等の補助ノードは不要。
    version = next_version_dir("data/overhead-camera/dataset/cube_only")
    output_dir = f"/data/overhead-camera/dataset/cube_only/{version}"
    print(f"出力先: {output_dir}")

    docker_exec('pkill -9 -f "cube_only.py"')
    time.sleep(1)

    result = docker_exec(
        f"{SETUP}; cd /workspace && pixi run cube_only "
        f"--headless --output-dir {output_dir} --num-frames {num_frames}"
    )
    print("\n".join((result.stdout or "").splitlines()[-10:]))

    print("=== 収集完了 ===")
    print(image_count(output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="method", required=True)

    picking = subparsers.add_parser("picking-session", help="vp-run-gtの自然なピック軌道を記録する")
    picking.add_argument("--camera", required=True, choices=["overhead-camera", "wrist-camera"])
    picking.add_argument("--iterations", type=int, default=40)
    picking.add_argument("--frames", type=int, default=8)

    jitter = subparsers.add_parser("jitter", help="手先カメラ向け、物体近傍のランダム位置から撮る")
    jitter.add_argument("--iterations", type=int, default=10)
    jitter.add_argument("--frames", type=int, default=30)

    cube_only = subparsers.add_parser("cube-only", help="俯瞰カメラ向け、孤立した合成シーンで生成する")
    cube_only.add_argument("--num-frames", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "picking-session":
        collect_picking_session(args.camera, args.iterations, args.frames)
    elif args.method == "jitter":
        collect_jitter(args.iterations, args.frames)
    elif args.method == "cube-only":
        collect_cube_only(args.num_frames)


if __name__ == "__main__":
    main()
