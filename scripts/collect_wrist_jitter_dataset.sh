#!/usr/bin/env bash
# collect_wrist_picking_session_dataset.shはvp-run-gtの自然なピック軌道をそのまま記録するため、
# 手先カメラがpanda_handからローカルX+0.05mオフセットして取り付けられている都合上、物体が
# 画像内の同じような位置(実測でlabels.jpgにより確認済み、左上寄り)にしか写らないデータに
# 偏ってしまう。このスクリプトはcollect_wrist_jitter_dataset.pyでアームを物体近傍の
# ランダムな相対位置・向きへ直接動かしながら撮ることで、物体が画像内の様々な位置・大きさで
# 写るデータを収集する。ピック動作自体は行わないため、picking_controller_nodeは使わない。
#
# 実行のたびにdata/wrist-camera/dataset/配下へ新しいバージョンディレクトリを採番する
# (過去の収集結果は消さずに積み上げる)。
set -uo pipefail

ITERATIONS="${1:-10}"
FRAMES_PER_RUN="${2:-30}"
# data/はdocker-compose.local.ymlで../../data:/dataとバインドマウントされているため、
# ホスト上のこのスクリプトから直接次バージョンを採番できる。
VERSION=$(bash "$(dirname "$0")/next_version_dir.sh" "data/wrist-camera/dataset")
OUTPUT_DIR="/data/wrist-camera/dataset/$VERSION"
X_MIN=0.388
X_MAX=0.612
Y_MIN=-0.073
Y_MAX=0.073

echo "出力先: $OUTPUT_DIR"

cleanup() {
    docker exec edge_container pkill -9 -f "run_simulation.py" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "vision_picking/lib/vision_picking/gt_tf_publisher_node" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "vision_picking/lib/vision_picking/camera_bridge_node" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "collect_wrist_jitter_dataset.py" >/dev/null 2>&1
    sleep 1
}

for i in $(seq 0 $((ITERATIONS - 1))); do
    cleanup
    X=$(python3 -c "import random; print(round(random.uniform($X_MIN,$X_MAX),3))")
    Y=$(python3 -c "import random; print(round(random.uniform($Y_MIN,$Y_MAX),3))")
    YAW=$(python3 -c "import random, math; print(round(random.uniform(-math.pi,math.pi),3))")
    START_INDEX=$((i * FRAMES_PER_RUN))
    echo "=== iteration $i/$((ITERATIONS - 1)): x=$X y=$Y yaw=$YAW start_index=$START_INDEX ==="

    docker exec -d edge_container bash -c "export OMNI_KIT_ACCEPT_EULA=YES TARGET_OBJECT_X=$X TARGET_OBJECT_Y=$Y TARGET_OBJECT_YAW=$YAW; source /workspace/install/setup.bash 2>/dev/null; cd /workspace && pixi run sim --headless" >/dev/null 2>&1

    ready=0
    for _ in $(seq 1 60); do
        if docker exec edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; timeout 5 ros2 topic echo /ground_truth/tf --once 2>/dev/null" 2>/dev/null | grep -q "target_object"; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "  sim起動タイムアウト、スキップ"
        continue
    fi

    docker exec -d edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; cd /workspace && pixi run gt_tf_publisher_node" >/dev/null 2>&1
    docker exec -d edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; cd /workspace && pixi run camera_bridge_node" >/dev/null 2>&1
    sleep 3

    docker exec edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; cd /workspace && timeout 120 pixi run collect_wrist_jitter_dataset --output-dir $OUTPUT_DIR --num-frames $FRAMES_PER_RUN --start-index $START_INDEX" 2>&1 | tail -10

    count=$(docker exec edge_container bash -c "ls $OUTPUT_DIR/images 2>/dev/null | wc -l")
    echo "  現在の総フレーム数: $count"
done

cleanup
echo "=== 収集完了 ==="
docker exec edge_container bash -c "ls $OUTPUT_DIR/images 2>/dev/null | wc -l"
