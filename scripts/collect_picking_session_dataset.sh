#!/usr/bin/env bash
# vp-run-gtでキューブをランダム位置に置きながらpicking_session.pyで学習用データを記録する。
# 1回のシム実行あたりのフレーム数を絞る代わりにイテレーション数(=ランダム配置のバリエーション数)を
# 増やす設計にしている。物体はピック中の一瞬しか姿勢が変化せず、1回のシム実行内で
# フレーム数を稼いでもほぼ同一姿勢の重複フレームが増えるだけで、姿勢の多様性には寄与しないため。
# 実行のたびにdata/overhead-camera/dataset/配下へ新しいバージョンディレクトリを採番する
# (過去の収集結果は消さずに積み上げる)。
set -uo pipefail

ITERATIONS="${1:-40}"
FRAMES_PER_RUN="${2:-8}"
# data/はdocker-compose.local.ymlで../../data:/dataとバインドマウントされているため、
# ホスト上のこのスクリプトから直接次バージョンを採番できる。
VERSION=$(bash "$(dirname "$0")/next_version_dir.sh" "data/overhead-camera/dataset")
OUTPUT_DIR="/data/overhead-camera/dataset/$VERSION"
# カメラは(0.5, 0.0, 0.8)固定・真下向きで、実測画角(fx=fy=1527px, キューブ上面までの
# 距離0.75m)は約31cm x 24cm。yaw回転時のキューブ最大到達距離(対角の半分、約3.5cm)と
# 1cmの安全マージンを差し引いた、中心が確実に画角内に収まる範囲が以下。
# これより広い範囲でランダム配置すると、画角外でフレームが1枚も撮れないイテレーションが多発する。
X_MIN=0.388
X_MAX=0.612
Y_MIN=-0.073
Y_MAX=0.073

echo "出力先: $OUTPUT_DIR"

cleanup() {
    docker exec edge_container pkill -9 -f "run_simulation.py" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "vision_picking/lib/vision_picking/gt_tf_publisher_node" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "vision_picking/lib/vision_picking/camera_bridge_node" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "vision_picking/lib/vision_picking/picking_controller_node" >/dev/null 2>&1
    docker exec edge_container pkill -9 -f "picking_session.py" >/dev/null 2>&1
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
        # ros2 topic hzは自分から終了しないため、timeout Nを付けても常にフルN秒待ってから
        # 強制終了される。準備できればすぐ1件返るtopic echo --onceの方がポーリングに向く。
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
    docker exec -d edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; cd /workspace && pixi run picking_session --output-dir $OUTPUT_DIR --max-frames $FRAMES_PER_RUN --start-index $START_INDEX --min-interval-sec 0.15" >/dev/null 2>&1
    sleep 1

    docker exec edge_container bash -c "source /workspace/install/setup.bash 2>/dev/null; cd /workspace && timeout 30 pixi run picking_controller_node" 2>&1 | tail -3

    sleep 2
    count=$(docker exec edge_container bash -c "ls $OUTPUT_DIR/images 2>/dev/null | wc -l")
    echo "  現在の総フレーム数: $count"
done

cleanup
echo "=== 収集完了 ==="
docker exec edge_container bash -c "ls $OUTPUT_DIR/images 2>/dev/null | wc -l"
