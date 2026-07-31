.PHONY: up build test build-test new-launch-pkg rviz rviz-flat

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

# ROS2: 各launchの起動
sim:
	$(MAKE) colcon CMD_RUN="pixi run sim"
