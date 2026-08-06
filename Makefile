.PHONY: up build new-launch-pkg dataset train dataset-split dataset-collect-overhead-picking-session dataset-collect-wrist-picking-session dataset-collect-wrist-jitter dataset-collect-overhead-cube-only model-promote vp-pose-estimation-overhead vp-run-yolo-latest-train vp-run-yolo-latest-models

COMPOSE_PJ_NAME    := vpicking
# WSL2かネイティブLinuxかでGPUパススルーの構成が別物になる(edge/docker/docker-compose.gpu-*.yml参照)
# ため、/proc/versionで自動判定してoverrideファイルを差し替える。
ifneq ($(shell grep -qi microsoft /proc/version 2>/dev/null && echo wsl),)
GPU_COMPOSE_FILE   := edge/docker/docker-compose.gpu-wsl.yml
else
GPU_COMPOSE_FILE   := edge/docker/docker-compose.gpu-native.yml
endif
# ホストUID/GIDをdocker-compose.local.ymlのビルド引数に渡し、
# コンテナ内の共有ボリューム生成物がroot所有になるのを防ぐ。
export HOST_UID      := $(shell id -u)
export HOST_GID      := $(shell id -g)
export HOST_USERNAME := $(shell id -un)
COMPOSE            := docker compose -f edge/docker/docker-compose.local.yml -f $(GPU_COMPOSE_FILE) -p $(COMPOSE_PJ_NAME)
RUN                := $(COMPOSE) run --rm --remove-orphans
EXEC               := $(COMPOSE) exec
ROS2_SERVICE       := edge
ROS2_CONTAINER     := edge_container
ROS2_WS            := /workspace
PKG                ?=
PKG_NAME           := $(if $(PKG),$(notdir $(PKG)),)
CMD_ROS2_WS_SOURCE := test -f $(ROS2_WS)/install/setup.bash	&& source $(ROS2_WS)/install/setup.bash || true

up:
	@test -f .env || cp .env.example .env
	$(COMPOSE) --env-file .env up -d
	@. ./.env 2>/dev/null; \

down:
	$(COMPOSE) --env-file .env down

build:
	$(COMPOSE) --env-file .env build
build-clean:
	$(COMPOSE) --env-file .env build --no-cache

new-launch-pkg:
	python3 scripts/create_launch_packag

login:
	$(EXEC) $(ROS2_SERVICE) bash -c \
	  "$(CMD_ROS2_WS_SOURCE) && bash"

nvidia-smi:
	$(EXEC) $(ROS2_SERVICE) bash -c nvidia-smi

colcon:
	$(EXEC) $(ROS2_SERVICE) bash -c \
	  "$(CMD_ROS2_WS_SOURCE) && \
	   cd $(ROS2_WS) && $(CMD_RUN)"

colcon-build:
	$(MAKE) colcon CMD_RUN="pixi run build"
colcon-build-clean:
	$(MAKE) colcon CMD_RUN="pixi run clean && pixi run build"
colcon-release-build:
	$(MAKE) colcon CMD_RUN="pixi run clean && pixi run release-build"

# ROS2: デバッグコマンド
WORLD_FRAME        := world
FROM               ?= $(WORLD_FRAME)
TO                 ?=

topic-info-ground-truth-tf:
	$(MAKE) colcon CMD_RUN="ros2 topic info /ground_truth/tf"
run-tf2-ros-tf2-echo:
ifeq ($(TO),)
	@echo "TOが未指定です。例: make run-tf2-ros-tf2-echo FROM=world TO=target_object"
	@exit 1
endif
	$(MAKE) colcon CMD_RUN="ros2 run tf2_ros tf2_echo $(FROM) $(TO)"

interface-show:
	$(MAKE) colcon CMD_RUN=" \
		ros2 interface show tf2_msgs/msg/TFMessage \
	"

# ROS2: 各node & launchの起動
sim:
	$(MAKE) colcon CMD_RUN="pixi run sim"
vp-gt-tf-publisher:
	$(MAKE) colcon CMD_RUN="pixi run gt_tf_publisher_node"
vp-picking-controller:
	$(MAKE) colcon CMD_RUN="pixi run picking_controller_node"
vp-camera-bridge:
	$(MAKE) colcon CMD_RUN="pixi run camera_bridge_node"
vp-pose-estimation-classical-cv:
	$(MAKE) colcon CMD_RUN="pixi run pose_estimation_node_classical_cv"
# Foxglove Studio(https://app.foxglove.dev)からws://localhost:8765に接続してカメラ画像・TF等を可視化する。
vp-foxglove:
	$(MAKE) colcon CMD_RUN="pixi run foxglove_bridge"

# edge/workspace/scripts/dataset/ 配下にあるスクリプトを対話式で選んで実行する。
dataset:
	@bash scripts/select_method.sh dataset

# ベースモデル(既定data/yolo/models/yolo11n-pose.pt)から
# data/<camera>/dataset/<script>/<version>/dataset.yamlでYOLO11-Poseを学習する
# (例: make train CAMERA=overhead-camera DATA=/data/overhead-camera/dataset/picking_session/v4/dataset.yaml)。
# CAMERA・DATAを省略すると対話式に選べる。由来の異なるデータセットを組み合わせたい場合は、
# 先にmake dataset-splitで結合したdataset.yamlを作ってからDATAで指定すること。
CAMERA ?=
DATA ?=
EPOCHS ?= 100
MODEL ?=
train:
	@if [ -z "$(CAMERA)" ] || [ -z "$(DATA)" ]; then \
		SEL=$$(bash scripts/select_train_target.sh "$(CAMERA)"); \
		CAM=$$(echo "$$SEL" | cut -d' ' -f1); \
		DAT=$$(echo "$$SEL" | cut -d' ' -f2); \
	else \
		CAM="$(CAMERA)"; DAT="$(DATA)"; \
	fi; \
	$(MAKE) colcon CMD_RUN="pixi run yolo_pose --camera $$CAM --data $$DAT --epochs $(EPOCHS) $(if $(MODEL),--model $(MODEL),)"

# データ収集後・学習前に1回実行し、data/<camera>/dataset/<script>/<version>のtrain/valを
# リークしない形に分割する
# (例: make dataset-split DATASET_DIR=/data/overhead-camera/dataset/picking_session/v1)。
# コンテナ内でdata/はワークスペース(/workspace)配下ではなく/data直下にマウントされているため、
# DATASET_DIRは/dataから始まる絶対パスで指定する必要がある。
# スペース区切りで複数指定すると、由来の異なるデータセットを結合した1つの学習データにできる
# (OUTPUT_DIRを省略すると各<script>を+で連結した名前の下に自動採番される)。
DATASET_DIR ?=
OUTPUT_DIR ?=
dataset-split:
	@if [ -z "$(DATASET_DIR)" ]; then echo "使い方: make dataset-split DATASET_DIR=/data/overhead-camera/dataset/picking_session/v1 [OUTPUT_DIR=...]" >&2; exit 1; fi
	$(MAKE) colcon CMD_RUN="pixi run split_dataset --dataset-dir $(DATASET_DIR) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)"

# scripts/collect_dataset.py(ホスト側、docker exec経由でedge_containerを操作する)が
# シムの起動〜ランダム配置〜記録〜クリーンアップの共通処理を担い、--methodで収集方式を
# 切り替える。事前にmake simを実行しておく必要はない(スクリプト自身がシムを都度起動する)。
# 実行のたびにdata/<camera>/dataset/<method>/配下へ新しいバージョンディレクトリを採番する。
ITERATIONS ?= 40
FRAMES     ?= 8
dataset-collect-overhead-picking-session:
	@python3 scripts/collect_dataset.py picking-session --camera overhead-camera --iterations $(ITERATIONS) --frames $(FRAMES)

# 俯瞰カメラ向けと同じ収集方式を手先カメラ(data/wrist-camera/dataset/picking_session/)向けに行う。
dataset-collect-wrist-picking-session:
	@python3 scripts/collect_dataset.py picking-session --camera wrist-camera --iterations $(ITERATIONS) --frames $(FRAMES)

# 上記の自然なピック軌道の記録だけだと、手先カメラの取り付けオフセットの影響で物体が
# 画像内の同じような位置にしか写らないデータに偏る(labels.jpgで実測確認済み)ため、
# アームを物体近傍のランダムな相対位置・向きへ直接動かしながら撮る方式で補う。
JITTER_ITERATIONS ?= 10
JITTER_FRAMES     ?= 30
dataset-collect-wrist-jitter:
	@python3 scripts/collect_dataset.py jitter --iterations $(JITTER_ITERATIONS) --frames $(JITTER_FRAMES)

# Pandaアームを含まない孤立した合成シーンでキューブのみのデータセットを生成する
# (data/overhead-camera/dataset/cube_only/へ採番)。他の収集ターゲットと違い自前で
# SimulationAppを起動する自己完結型のため、事前にmake simを実行しておく必要はない。
CUBE_ONLY_FRAMES ?= 50
dataset-collect-overhead-cube-only:
	@python3 scripts/collect_dataset.py cube-only --num-frames $(CUBE_ONLY_FRAMES)

# 気に入った学習結果を data/<camera>/train-models/<VER>/(ローカルのみ)から
# data/<camera>/models/(git管理下・push対象)へ昇格させる。data/<camera>/models/側は
# 「pushしたモデルの通し番号」として独立に採番する(train-models側の番号をそのまま使い回さない)。
# train-models配下はdata/<camera>/train-models/<script>/vNとネストしているため、
# VERは"<script>/vN"の形式で指定する(例: picking_session/v7、jitter/v3)。
# CAMERA・VERを省略すると、scripts/select_promote_target.pyで対話式に選べる。
model-promote:
	@if [ -z "$(CAMERA)" ] || [ -z "$(VER)" ]; then \
		SEL=$$(python3 scripts/select_promote_target.py "$(CAMERA)"); \
		CAM=$$(echo "$$SEL" | cut -d' ' -f1); \
		V=$$(echo "$$SEL" | cut -d' ' -f2); \
	else \
		CAM="$(CAMERA)"; V="$(VER)"; \
	fi; \
	if [ ! -f "data/$$CAM/train-models/$$V/best.pt" ]; then echo "data/$$CAM/train-models/$$V/best.pt が見つかりません" >&2; exit 1; fi; \
	NEW_V=$$(bash scripts/next_version_dir.sh "data/$$CAM/models"); \
	mkdir -p "data/$$CAM/models/$$NEW_V" && \
	cp "data/$$CAM/train-models/$$V/best.pt" "data/$$CAM/models/$$NEW_V/best.pt" && \
	cp "data/$$CAM/train-models/$$V/object_3d_keypoints.json" "data/$$CAM/models/$$NEW_V/object_3d_keypoints.json" && \
	echo "data/$$CAM/train-models/$$V -> data/$$CAM/models/$$NEW_V に昇格しました"

# VERは"v1"(data/overhead-camera/models/配下、git管理・push済み)または
# "train:<script>/v7"(data/overhead-camera/train-models/<script>/配下、ローカルのみ・push前)
# の形式で指定する(例: make vp-pose-estimation-overhead VER=v1、VER=train:picking_session/v7)。
# 未指定時はscripts/select_version.shで対話式に選ばせる。
# このターゲットは俯瞰カメラ単独運用(CAMERA_NAMESPACE既定値)向け。
VER ?=
vp-pose-estimation-overhead:
	@V="$(VER)"; \
	if [ -z "$$V" ]; then V=$$(bash scripts/select_version.sh "" overhead-camera); fi; \
	$(MAKE) colcon CMD_RUN="MODEL_VERSION=$$V pixi run pose_estimation_node"

# ROS2: Vision Pickingの起動
# ※ 先に`make sim`を実行してIsaac Simを起動。
#
# vp-run-gt: 座標をそのまま使ったピッキング
# vp-run-cv: Depthカメラの画像から物体姿勢を推定してピッキング(古典的CV版)
# vp-run-yolo: RGB画像からYOLO11-Pose+PnPで物体姿勢を推定してピッキング(本番版)
vp-run-gt:
	@$(MAKE) vp-gt-tf-publisher EXEC="$(COMPOSE) exec -T" & \
	trap '$(EXEC) $(ROS2_SERVICE) bash -c "pkill -f vision_picking/lib/vision_picking/gt_tf_publisher_node" >/dev/null 2>&1 || true' EXIT INT TERM; \
	$(MAKE) vp-picking-controller
vp-run-cv:
	@$(MAKE) vp-camera-bridge EXEC="$(COMPOSE) exec -T" & \
	$(MAKE) vp-pose-estimation-classical-cv EXEC="$(COMPOSE) exec -T" & \
	trap '$(EXEC) $(ROS2_SERVICE) bash -c "pkill -f vision_picking/lib/vision_picking/camera_bridge_node; pkill -f vision_picking/lib/vision_picking/pose_estimation_node_classical_cv" >/dev/null 2>&1 || true' EXIT INT TERM; \
	$(MAKE) vp-picking-controller
# vp-run-yolo: 俯瞰カメラ(粗検出、VER_OVERHEAD)・手先カメラ(精緻検出、VER_WRIST)の
# 2つのpose_estimation_nodeインスタンスを同時起動する(pose_estimation_node.py自体は
# CAMERA_NAMESPACE/TARGET_FRAME_OUT環境変数だけでどちらの用途にもそのまま使える)。
VER_OVERHEAD ?=
VER_WRIST    ?=
vp-run-yolo:
	@VO="$(VER_OVERHEAD)"; \
	if [ -z "$$VO" ]; then VO=$$(bash scripts/select_version.sh "俯瞰カメラ(粗検出)" overhead-camera); fi; \
	VW="$(VER_WRIST)"; \
	if [ -z "$$VW" ]; then VW=$$(bash scripts/select_version.sh "手先カメラ(精緻検出)" wrist-camera); fi; \
	$(MAKE) vp-camera-bridge EXEC="$(COMPOSE) exec -T" & \
	$(MAKE) colcon CMD_RUN="MODEL_VERSION=$$VO CAMERA_NAMESPACE=camera TARGET_FRAME_OUT=target_object_coarse pixi run pose_estimation_node" EXEC="$(COMPOSE) exec -T" & \
	$(MAKE) colcon CMD_RUN="MODEL_VERSION=$$VW CAMERA_NAMESPACE=wrist_camera TARGET_FRAME_OUT=target_object_fine pixi run pose_estimation_node" EXEC="$(COMPOSE) exec -T" & \
	trap '$(COMPOSE) exec -T $(ROS2_SERVICE) bash -c "pkill -f vision_picking/lib/vision_picking/camera_bridge_node; pkill -f vision_picking/lib/vision_picking/pose_estimation_node$$" >/dev/null 2>&1 || true' EXIT INT TERM; \
	$(MAKE) vp-picking-controller

# vp-run-yolo-latest-train/vp-run-yolo-latest-models: select_version.shの対話選択を省略し、
# data/<camera>/train-models/(ローカルのみ・push前)またはdata/<camera>/models/(git管理・
# push済み)それぞれの最新バージョンを俯瞰・手先の両方に自動で使ってvp-run-yoloを起動する。
vp-run-yolo-latest-train:
	@VO=$$(bash scripts/latest_version_dir.sh data/overhead-camera/train-models); \
	VW=$$(bash scripts/latest_version_dir.sh data/wrist-camera/train-models); \
	echo "俯瞰: train:$$VO / 手先: train:$$VW"; \
	$(MAKE) vp-run-yolo VER_OVERHEAD=train:$$VO VER_WRIST=train:$$VW

vp-run-yolo-latest-models:
	@VO=$$(bash scripts/latest_version_dir.sh data/overhead-camera/models); \
	VW=$$(bash scripts/latest_version_dir.sh data/wrist-camera/models); \
	echo "俯瞰: $$VO / 手先: $$VW"; \
	$(MAKE) vp-run-yolo VER_OVERHEAD=$$VO VER_WRIST=$$VW
