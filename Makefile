.PHONY: up build new-launch-pkg dataset train dataset-split dataset-collect-picking-session dataset-collect-wrist-picking-session dataset-collect-wrist-jitter model-promote vp-run-yolo-latest-train vp-run-yolo-latest-models

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

# edge/workspace/scripts/dataset/ または train/ 配下にあるスクリプトを対話式で選んで実行する。
dataset:
	@bash scripts/select_method.sh dataset
train:
	@bash scripts/select_method.sh train

# データ収集後・学習前に1回実行し、data/<camera>/dataset/<version>のtrain/valを
# リークしない形に分割する(例: make dataset-split DATASET_DIR=data/overhead-camera/dataset/v1)。
DATASET_DIR ?=
dataset-split:
	@if [ -z "$(DATASET_DIR)" ]; then echo "使い方: make dataset-split DATASET_DIR=data/overhead-camera/dataset/v1" >&2; exit 1; fi
	$(MAKE) colcon CMD_RUN="pixi run split_dataset --dataset-dir $(DATASET_DIR)"

# make simでシムを起動する代わりに、シムの起動〜ランダム配置〜1回のピック実行〜記録を
# 指定回数繰り返す。事前にmake simを実行しておく必要はない(このターゲット自身がシムを都度起動する)。
# 実行のたびにdata/overhead-camera/dataset/配下へ新しいバージョンディレクトリを採番する。
ITERATIONS ?= 40
FRAMES     ?= 8
dataset-collect-picking-session:
	@bash scripts/collect_picking_session_dataset.sh $(ITERATIONS) $(FRAMES)

# 俯瞰カメラ向けと同じ収集方式を手先カメラ(data/wrist-camera/dataset/)向けに行う。
dataset-collect-wrist-picking-session:
	@bash scripts/collect_wrist_picking_session_dataset.sh $(ITERATIONS) $(FRAMES)

# 上記の自然なピック軌道の記録だけだと、手先カメラの取り付けオフセットの影響で物体が
# 画像内の同じような位置にしか写らないデータに偏る(labels.jpgで実測確認済み)ため、
# アームを物体近傍のランダムな相対位置・向きへ直接動かしながら撮る方式で補う。
JITTER_ITERATIONS ?= 10
JITTER_FRAMES     ?= 30
dataset-collect-wrist-jitter:
	@bash scripts/collect_wrist_jitter_dataset.sh $(JITTER_ITERATIONS) $(JITTER_FRAMES)

# 気に入った学習結果を data/<camera>/train-models/<VER>/(ローカルのみ)から
# data/<camera>/models/(git管理下・push対象)へ昇格させる。data/<camera>/models/側は
# 「pushしたモデルの通し番号」として独立に採番する(train-models側の番号をそのまま使い回さない)。
model-promote:
	@CAM="$(CAMERA)"; \
	V="$(VER)"; \
	if [ -z "$$CAM" ] || [ -z "$$V" ]; then echo "使い方: make model-promote CAMERA=wrist-camera VER=v7" >&2; exit 1; fi; \
	if [ ! -f "data/$$CAM/train-models/$$V/best.pt" ]; then echo "data/$$CAM/train-models/$$V/best.pt が見つかりません" >&2; exit 1; fi; \
	NEW_V=$$(bash scripts/next_version_dir.sh "data/$$CAM/models"); \
	mkdir -p "data/$$CAM/models/$$NEW_V" && \
	cp "data/$$CAM/train-models/$$V/best.pt" "data/$$CAM/models/$$NEW_V/best.pt" && \
	cp "data/$$CAM/train-models/$$V/object_3d_keypoints.json" "data/$$CAM/models/$$NEW_V/object_3d_keypoints.json" && \
	echo "data/$$CAM/train-models/$$V -> data/$$CAM/models/$$NEW_V に昇格しました"

# VERは"v1"(data/overhead-camera/models/配下、git管理・push済み)または"train:v7"
# (data/overhead-camera/train-models/配下、ローカルのみ・push前)の形式で指定する
# (例: make vp-pose-estimation VER=v1)。未指定時はscripts/select_version.shで対話式に選ばせる。
# このターゲットは俯瞰カメラ単独運用(CAMERA_NAMESPACE既定値)向け。
VER ?=
vp-pose-estimation:
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
# vp-run-yolo: 俯瞰カメラ(粗検出、VER_COARSE)・手先カメラ(精緻検出、VER_FINE)の
# 2つのpose_estimation_nodeインスタンスを同時起動する(pose_estimation_node.py自体は
# CAMERA_NAMESPACE/TARGET_FRAME_OUT環境変数だけでどちらの用途にもそのまま使える)。
VER_COARSE ?=
VER_FINE   ?=
vp-run-yolo:
	@VC="$(VER_COARSE)"; \
	if [ -z "$$VC" ]; then VC=$$(bash scripts/select_version.sh "俯瞰カメラ(粗検出)" overhead-camera); fi; \
	VF="$(VER_FINE)"; \
	if [ -z "$$VF" ]; then VF=$$(bash scripts/select_version.sh "手先カメラ(精緻検出)" wrist-camera); fi; \
	$(MAKE) vp-camera-bridge EXEC="$(COMPOSE) exec -T" & \
	$(MAKE) colcon CMD_RUN="MODEL_VERSION=$$VC CAMERA_NAMESPACE=camera TARGET_FRAME_OUT=target_object_coarse pixi run pose_estimation_node" EXEC="$(COMPOSE) exec -T" & \
	$(MAKE) colcon CMD_RUN="MODEL_VERSION=$$VF CAMERA_NAMESPACE=wrist_camera TARGET_FRAME_OUT=target_object_fine pixi run pose_estimation_node" EXEC="$(COMPOSE) exec -T" & \
	trap '$(COMPOSE) exec -T $(ROS2_SERVICE) bash -c "pkill -f vision_picking/lib/vision_picking/camera_bridge_node; pkill -f vision_picking/lib/vision_picking/pose_estimation_node$$" >/dev/null 2>&1 || true' EXIT INT TERM; \
	$(MAKE) vp-picking-controller

# vp-run-yolo-latest-train/vp-run-yolo-latest-models: select_version.shの対話選択を省略し、
# data/<camera>/train-models/(ローカルのみ・push前)またはdata/<camera>/models/(git管理・
# push済み)それぞれの最新バージョンを俯瞰・手先の両方に自動で使ってvp-run-yoloを起動する。
vp-run-yolo-latest-train:
	@VC=$$(bash scripts/latest_version_dir.sh data/overhead-camera/train-models); \
	VF=$$(bash scripts/latest_version_dir.sh data/wrist-camera/train-models); \
	echo "俯瞰: train:$$VC / 手先: train:$$VF"; \
	$(MAKE) vp-run-yolo VER_COARSE=train:$$VC VER_FINE=train:$$VF

vp-run-yolo-latest-models:
	@VC=$$(bash scripts/latest_version_dir.sh data/overhead-camera/models); \
	VF=$$(bash scripts/latest_version_dir.sh data/wrist-camera/models); \
	echo "俯瞰: $$VC / 手先: $$VF"; \
	$(MAKE) vp-run-yolo VER_COARSE=$$VC VER_FINE=$$VF
