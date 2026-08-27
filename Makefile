# make file inspired by https://roborovsky-racers.github.io/RoborovskyNote/
SHELL := /bin/bash

.PHONY: autoware-build autoware-vehicle autoware-simulator autoware-request-initialpose autoware-request-control  awsim-request-start awsim-request-reset autoware-driver-zenoh autoware-driver-zenoh-rosbag \
	simulator dev dev2 dev3 dev4 lidar-rl-awsim lidar-rl-awsim4 \
	lidar-rl-request-control lidar-rl-request-control4 driver zenoh download \
	rviz2 down down_all ps autoware-attach autoware-bash eval

# Used by docker-compose.yml for build/eval artifact ownership.
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)
export HOST_UID HOST_GID

# LiDAR-only RL uses a standalone Python/uv image. Set LIDAR_RL_GPU=1 to add
# the NVIDIA overlay; CPU is the portable default (including macOS smoke use).
LIDAR_RL_DIR := aichallenge/ml_workspace/lidar_racing_rl
LIDAR_RL_GPU ?= 0
LIDAR_RL_COMPOSE = docker compose --project-directory $(LIDAR_RL_DIR) \
	-f $(LIDAR_RL_DIR)/compose.yaml \
	$(if $(filter 1 true yes,$(LIDAR_RL_GPU)),-f $(LIDAR_RL_DIR)/compose.gpu.yaml)
LIDAR_RL_RUN = $(LIDAR_RL_COMPOSE) run --rm --no-deps lidar-rl
LIDAR_RL_ARGS ?=
# The path dependency is optional in pyproject.toml, but every executable
# environment entry point needs it. CUDA remains opt-in with the GPU overlay.
LIDAR_RL_UV_EXTRAS = --extra f1tenth \
	$(if $(filter 1 true yes,$(LIDAR_RL_GPU)),--extra cuda)
LIDAR_RL_SYNC_ARGS ?= $(LIDAR_RL_UV_EXTRAS)
LIDAR_RL_RUN_ARGS ?= $(LIDAR_RL_UV_EXTRAS)
LIDAR_RL_EXPORT_RUN_ARGS ?= $(LIDAR_RL_UV_EXTRAS) --extra export
# The full pytest suite includes Flax-to-PyTorch parity and bundle tests.
LIDAR_RL_TEST_RUN_ARGS ?= $(LIDAR_RL_EXPORT_RUN_ARGS)
# Stop host shell's ROS_DOMAIN_ID from overriding .env via compose interpolation,
# but still honor an explicit `make foo ROS_DOMAIN_ID=N` command-line override.
unexport ROS_DOMAIN_ID
ifeq ($(origin ROS_DOMAIN_ID),command line)
export ROS_DOMAIN_ID
endif

TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
LOG_DIR := /output/$(TIMESTAMP)

# make simulator-<mode>: <mode> は simulator_scripts/*.sh のファイル名
SIM_MODES := $(notdir $(basename $(wildcard aichallenge/simulator_scripts/*.sh)))
# dev<N>（車両数）/ gate<N>（テスト番号）は run_simulator.bash が展開するエイリアス
SIM_MODES += dev2 dev3 dev4 gate1 gate2 gate3
.PHONY: $(addprefix simulator-,$(SIM_MODES))
$(addprefix simulator-,$(SIM_MODES)): simulator-%:
	@$(MAKE) simulator SIM_MODE=$*

# autowareのbuildのみ
autoware-build:
	docker compose run -T --rm --no-deps autoware-build

# run autoware for vehicle
autoware-vehicle:
	@echo "Start Autoware for Vehicle"
	@echo "Log dir: .$(LOG_DIR)"
	LOG_DIR=$(LOG_DIR) RUN_MODE=vehicle docker compose up -d autoware

# run autoware for simulator
autoware-simulator:
	@echo "Start Autoware for AWSIM"
	@echo "Log dir: .$(LOG_DIR)"
	LOG_DIR=$(LOG_DIR) RUN_MODE=awsim docker compose up -d autoware

# autoware command service use ROS_DOMAIN_ID from .env
autoware-request-initialpose:
	CMD="ros2 service call /set_initial_pose std_srvs/srv/Trigger '{}'" docker compose run --rm --no-deps autoware-command

autoware-request-control:
	CMD="ros2 topic pub -1 /awsim/control_mode_request_topic std_msgs/msg/Bool '{data: true}'" docker compose run --rm --no-deps autoware-command

# awsim admin service use ROS_DOMAIN_ID 0
awsim-request-start:
	CMD="env ROS_DOMAIN_ID=0 ros2 topic pub -1 /admin/awsim/start std_msgs/msg/Bool '{data: true}'" docker compose run --rm --no-deps autoware-command

awsim-request-reset:
	CMD="env ROS_DOMAIN_ID=0 ros2 topic pub -1 /admin/awsim/reset std_msgs/msg/Empty '{}'" docker compose run --rm --no-deps autoware-command

# run simulator (docker compose up -d simulator)
simulator:
	@echo "Start AWSIM (SIM_MODE=$(SIM_MODE))"
	@echo "Log dir: .$(LOG_DIR)"
	LOG_DIR=$(LOG_DIR) SIM_MODE="$(SIM_MODE)" ROS_DOMAIN_ID=0 docker compose up -d simulator

# racing kart (docker compose up -d driver)
driver:
	docker compose up -d driver

# zenoh (docker compose up -d zenoh)
zenoh:
	docker compose up -d zenoh

dev: SIM_MODE := dev
dev: simulator autoware-simulator
	@echo "Start dev simulation (AWSIM + Autoware)"
	@echo "To stop: make down  (docker compose down --remove-orphans)"

dev2: SIM_MODE := dev2
dev3: SIM_MODE := dev3
dev4: SIM_MODE := dev4
dev2 dev3 dev4: simulator
	@N=$(@:dev%=%); \
	echo "Start $$N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$$N via docker compose -p)"; \
	for p in $$(seq 1 $$N); do LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=$$p docker compose -p $$p up -d autoware; done; \
	echo "To Stop: make down"

# AWSIM transfer and ROS inference stay in the existing AI Challenge image.
# The model may be absent during bring-up; the controller then publishes only
# its fail-safe command. Bulk F1TENTH/JAX work uses the standalone ML image.
lidar-rl-awsim:
	LOG_DIR=$(LOG_DIR) SIM_MODE=lidar-rl ROS_DOMAIN_ID=0 \
		LIDAR_RL_VEHICLES=1 docker compose up -d simulator
	LOG_DIR=$(LOG_DIR) RUN_MODE=awsim ROS_DOMAIN_ID=1 \
		CONTROL_METHOD=lidar_racing docker compose up -d autoware
	@echo "Started LiDAR-only AWSIM and lidar_racing_controller (ROS domain 1)"

lidar-rl-awsim4:
	LOG_DIR=$(LOG_DIR) SIM_MODE=lidar-rl ROS_DOMAIN_ID=0 \
		LIDAR_RL_VEHICLES=4 docker compose up -d simulator
	@for p in 1 2 3 4; do \
		LOG_DIR=$(LOG_DIR) RUN_MODE=awsim ROS_DOMAIN_ID=$$p \
			CONTROL_METHOD=lidar_racing docker compose -p $$p up -d autoware; \
	done
	@echo "Started four LiDAR-only AWSIM domains with lidar_racing_controller"

# Run these after the corresponding Autoware containers have discovered the
# AWSIM control-mode request topic. Keeping authorization explicit avoids a
# startup sleep whose correct duration depends on the host.
lidar-rl-request-control:
	ROS_DOMAIN_ID=1 CMD="ros2 topic pub -1 /awsim/control_mode_request_topic std_msgs/msg/Bool '{data: true}'" \
		docker compose run --rm --no-deps autoware-command

lidar-rl-request-control4:
	@for p in 1 2 3 4; do \
		echo "Request AWSIM control on ROS domain $$p"; \
		ROS_DOMAIN_ID=$$p \
			CMD="ros2 topic pub -1 /awsim/control_mode_request_topic std_msgs/msg/Bool '{data: true}'" \
			docker compose -p $$p run --rm --no-deps autoware-command; \
	done

gate1: SIM_MODE := gate1
gate2: SIM_MODE := gate2
gate3: SIM_MODE := gate3
gate1 gate2 gate3: simulator autoware-simulator
	@echo "Start safety gate simulation (AWSIM + Autoware)"
	@echo "To stop: make down  (docker compose down --remove-orphans)"

eval:
	@echo "Start evaluation simulation (AWSIM + Autoware)"
	docker compose up -d autoware-simulator-evaluation
	$(MAKE) awsim-request-start
	@echo "To stop: make down  (docker compose down --remove-orphans)"

# remote operation (docker compose up -d rviz2)
rviz2:
	docker compose stop rviz2
	docker compose up -d rviz2

# driver + autoware + zenoh
autoware-driver-zenoh:
	LOG_DIR=$(LOG_DIR) RUN_MODE=vehicle docker compose up -d driver autoware
	sleep 15
	LOG_DIR=$(LOG_DIR) docker compose up -d zenoh

# driver + autoware + all-topic rosbag + zenoh
autoware-driver-zenoh-rosbag:
	LOG_DIR=$(LOG_DIR) RUN_MODE=vehicle docker compose up -d driver autoware rosbag
	sleep 15
	LOG_DIR=$(LOG_DIR) docker compose up -d zenoh

down:
	@for p in 1 2 3 4; do docker compose -p $$p down --remove-orphans; done
	@docker compose down --remove-orphans

down_all:
	sudo docker ps -aq | xargs -r sudo docker rm -f

ps:
	@docker compose ps
	@for p in 1 2 3 4; do \
		out=$$(docker compose -p $$p ps --format '{{.Name}}\t{{.Service}}\t{{.Status}}' 2>/dev/null); \
		if [ -n "$$out" ]; then \
			echo "--- project=$$p ---"; \
			echo "$$out"; \
		fi; \
	done

autoware-attach:
	@./docker_exec.sh

autoware-bash:
	CMD="bash --rcfile /etc/skel/.bashrc -i" docker compose run --rm --no-deps autoware-command

.PHONY: lidar-rl-static lidar-rl-setup lidar-rl-test lidar-rl-benchmark \
	lidar-rl-train-step1 lidar-rl-train-step2 lidar-rl-eval lidar-rl-export \
	lidar-rl-install-policy

LIDAR_RL_BUNDLE ?= $(LIDAR_RL_DIR)/exported
LIDAR_RL_INSTALL_ARGS ?=

lidar-rl-static:
	PYTHONDONTWRITEBYTECODE=1 python3 $(LIDAR_RL_DIR)/scripts/check_source_contract.py
	git diff --check
	git diff --cached --check
	git -C $(LIDAR_RL_DIR)/repos/f1tenth_gym_jax diff --check

lidar-rl-setup:
	$(LIDAR_RL_COMPOSE) build lidar-rl
	$(LIDAR_RL_RUN) uv sync $(LIDAR_RL_SYNC_ARGS)

lidar-rl-test:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_TEST_RUN_ARGS) python -m pytest $(LIDAR_RL_ARGS)

lidar-rl-benchmark:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_RUN_ARGS) python scripts/benchmark_env.py $(LIDAR_RL_ARGS)

lidar-rl-train-step1:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_RUN_ARGS) python scripts/train.py \
		--config-name step1_single_vehicle $(LIDAR_RL_ARGS)

lidar-rl-train-step2:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_RUN_ARGS) python scripts/train.py \
		--config-name step2_four_vehicle $(LIDAR_RL_ARGS)

lidar-rl-eval:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_RUN_ARGS) python scripts/evaluate.py $(LIDAR_RL_ARGS)

lidar-rl-export:
	$(LIDAR_RL_RUN) uv run --frozen $(LIDAR_RL_EXPORT_RUN_ARGS) python scripts/export_policy.py $(LIDAR_RL_ARGS)

lidar-rl-install-policy:
	python3 $(LIDAR_RL_DIR)/scripts/install_policy_bundle.py \
		--bundle $(LIDAR_RL_BUNDLE) $(LIDAR_RL_INSTALL_ARGS)

# Download submission data by asking for credentials interactively
# Usage:
#   make download [SUBMISSION_ID=<id>]
# Usage (Only Admins):
#   make download [USER_ID=<id>] [SUBMISSION_ID=<id>]
download:
	@if [ -n "$(USER_ID)" ]; then \
		if [ -n "$(SUBMISSION_ID)" ]; then \
			vehicle/download_submission.sh --output aichallenge/workspace/src/ --user-id $(USER_ID) --submission-id $(SUBMISSION_ID); \
		else \
			vehicle/download_submission.sh --output aichallenge/workspace/src/ --user-id $(USER_ID); \
		fi; \
	else \
		if [ -n "$(SUBMISSION_ID)" ]; then \
			vehicle/download_submission.sh --output aichallenge/workspace/src/ --submission-id $(SUBMISSION_ID); \
		else \
			vehicle/download_submission.sh --output aichallenge/workspace/src/; \
		fi; \
	fi
