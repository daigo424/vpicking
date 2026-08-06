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
# 画角内に収まる安全範囲は、カメラからキューブ上面までの実効距離(pinholeモデル)から
# yaw回転時のキューブ最大到達距離(対角の半分、約3.5cm)と1cmの安全マージンを差し引いて
# 計算する。aperture/focal_lengthの値・計算式・TABLE_HEIGHT_Mはrun_simulation.py/
# common/target_object_shape.pyと揃えること(画角自体を変えたらここも合わせて変える必要がある)。
TABLE_HEIGHT_M=0.30
CUBE_TOP_OFFSET_M=0.05
DIAGONAL_MARGIN_M=0.045
APERTURE_X_MM=2.0955
APERTURE_Y_MM=1.52908
FOCAL_LENGTH_MM=2.5
# 検出モデルは学習データに含まれる距離のキューブの見かけサイズにしか対応できないため、
# カメラ高さを1点に固定したまま学習データを集めると、別の距離では検出が破綻する。
# イテレーションごとに高さ自体もランダム化し、学習データに距離の多様性を持たせる。
CAMERA_HEIGHT_MIN=0.6
CAMERA_HEIGHT_MAX=1.0

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
    CAMERA_HEIGHT=$(python3 -c "import random; print(round(random.uniform($CAMERA_HEIGHT_MIN,$CAMERA_HEIGHT_MAX),3))")
    read -r X_MIN X_MAX Y_MIN Y_MAX <<< "$(python3 -c "
distance = $CAMERA_HEIGHT - $TABLE_HEIGHT_M - $CUBE_TOP_OFFSET_M
half_x = distance * $APERTURE_X_MM / (2 * $FOCAL_LENGTH_MM) - $DIAGONAL_MARGIN_M
half_y = distance * $APERTURE_Y_MM / (2 * $FOCAL_LENGTH_MM) - $DIAGONAL_MARGIN_M
print(round(0.5 - half_x, 3), round(0.5 + half_x, 3), round(-half_y, 3), round(half_y, 3))
")"
    X=$(python3 -c "import random; print(round(random.uniform($X_MIN,$X_MAX),3))")
    Y=$(python3 -c "import random; print(round(random.uniform($Y_MIN,$Y_MAX),3))")
    YAW=$(python3 -c "import random, math; print(round(random.uniform(-math.pi,math.pi),3))")
    START_INDEX=$((i * FRAMES_PER_RUN))
    echo "=== iteration $i/$((ITERATIONS - 1)): camera_height=$CAMERA_HEIGHT x=$X y=$Y yaw=$YAW start_index=$START_INDEX ==="

    docker exec -d edge_container bash -c "export OMNI_KIT_ACCEPT_EULA=YES CAMERA_HEIGHT_M=$CAMERA_HEIGHT TARGET_OBJECT_X=$X TARGET_OBJECT_Y=$Y TARGET_OBJECT_YAW=$YAW; source /workspace/install/setup.bash 2>/dev/null; cd /workspace && pixi run sim --headless" >/dev/null 2>&1

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
