# LiDAR Racing RL

2D LiDARだけを方策入力として、自動運転AIチャレンジのレーシングカートを制御するためのサブプロジェクトです。F1TENTH Gym JAX上で学習し、最終ActorをPyTorchへ変換してAWSIM用ROS 2パッケージへ渡す構成です。

## 現在の状態

環境・センサ・NPC、Ego-only SAC、決定論評価、モデル変換、ROS推論のコード経路は実装済みです。ただし、依存環境を含む実行検証と学習済みモデルの生成はまだ行っていません。「コードが存在すること」と「Docker上で学習・走行を確認済みであること」を区別してください。

実装済み:

- LiDAR-only環境wrapperの単一・batch `reset` / `step`、Ego単位の終了・自動reset
- 360 beam canonical scan、range / validity分離、4 frame stack、action変換、報酬・終了判定
- Ray–OBBと動的車両LiDAR、reset sampler、scan corruption設定境界
- JAX Pure Pursuit、NPC縦制御、エピソード単位のNPC randomization
- 車体幅を考慮したセンターライン基準、追い抜きヒステリシス、相対進行・安全接触・停滞報酬
- Pure Pursuit教師による単独rolloutと、64環境・4車両・360 beamの性能benchmark入口
- Flax Gaussian Actor、Twin Critic、automatic entropy tuning、Polyak更新、Ego-only JAX ring Replay Buffer
- SAC collector、NaN検出、累積進捗付きatomic checkpoint、warm restart、実験設定・root/fork SHA・dirty diffの記録
- checkpointを読む決定論評価、追従・追い抜き指標集計、Flax→PyTorch変換・`1e-5` parity gate・Manifest生成
- export bundleのchecksumを確認してROS 2提出パッケージへ配置するinstall入口
- AWSIM LaserScan rosbag解析と、ROS 2側の前処理・PyTorch runtime・安全停止・launch境界
- Unit / contract testコード

`scripts/train.py`、`scripts/evaluate.py`、`scripts/export_policy.py`は`--dry-run`ならJAXやPyTorchを初期化せずに設定・入出力契約だけを確認し、通常実行ではそれぞれ学習・評価・変換を行います。

未検証・blocker:

- `.gitmodules`とfork submoduleは登録済みです。forkの`aichallenge/dynamic-lidar`にはarray step/reset、LiDAR hook、scan-only観測、コース境界APIの変更がありますが、変更commitと親側gitlink更新はまだ行っていません。
- 現在表示されるsubmodule SHA `1b4eb3f5161756bb925987753b965b549097742f`は変更前の基点です。fork変更をreview・commit・forkへpushした後の新しい完全SHAを親側で固定する必要があります。
- `uv.lock`は未生成です。fork commitを固定した後、Ubuntu側の専用Dockerで初回setupを行って生成・reviewします。
- JAX初期化、依存test、Docker build、F1TENTH rollout、AWSIM end-to-end推論は未実行です。
- 学習済みweightは同梱していないため、ROS controllerはmodel未配置時に安全停止します。
- AWSIM車両応答の同定値とLiDAR corruption値は、実測artifactがないため未校正です。
- 操舵action上限はrepo内の車両情報に合わせて暫定`±0.64 rad`へ統一しています。
  AWSIM実測で狭めることはできますが、根拠なく広げないでください。
- AWSIM上のonline学習・fine-tuning経路は未実装です。現在のStep 3はF1TENTHで学習した
  Actorのexport、ROS 2推論、AWSIM評価までを対象にします。
- 既存AI Challenge側のbase imageはroot `Dockerfile`で`humble-latest`を参照しており、
  digest固定は未実施です。sealed評価の完全再現にはUbuntu側で採用digestを記録・固定する
  追加作業が必要です。
- Step 2の追い抜き報酬と指標はsource上で接続済みです。curriculum／opponent poolは定義・offline検証interfaceまで実装していますが、学習ループ統合は未完了のため既定設定で明示的に無効です。

## 情報利用境界

- ActorとCriticの入力は、現在・過去のLiDAR range、validity mask、LiDARから得た局所フィルタ結果だけです。
- GT位置、Frenet座標、地図上の自己位置、車両状態、前回行動はActor・Criticへ渡しません。
- GT情報は報酬、終了判定、NPC、初期配置、教師方策、評価、センサ生成に限って使用します。
- 学習transitionは各環境の`agent_0`だけをReplay Bufferへ保存する契約です。
- AWSIM推論側は`aichallenge/workspace/src/aichallenge_submit/lidar_racing_controller/`に分離し、JAXを持ち込みません。

## Docker境界

用途によって実行環境を分けます。

| 実行環境 | 対象 |
|---|---|
| 既存AI Challenge Docker | AWSIMのLiDAR・車両応答計測、教師データ生成、ROS 2 PyTorch推論、sealed評価 |
| `lidar_racing_rl`専用Docker | F1TENTH/JAXのoffline学習、環境benchmark、unit test、設定検証、AWSIMへ接続しないモデル変換 |

専用Dockerは既存Autoware imageを継承せず、リポジトリrootを`/workspace`へbind mountし、`.venv`とuv cacheをnamed volumeへ分離します。CPUが既定で、NVIDIA環境だけ`compose.gpu.yaml`を追加します。詳細は[Docker境界ADR](../../../docs/decisions/e2e-lidar-rl-docker-boundary.md)を参照してください。

## ディレクトリ

```text
lidar_racing_rl/
├── assets/calibration/       # AWSIM実測統計。推測値で埋めない
├── configs/                  # Hydra/OmegaConf設定
├── docker/                   # Python 3.12 + uv専用image
├── repos/                    # fork submoduleと参照リポジトリの記録
├── scripts/                  # benchmark、rollout、学習・評価・変換、AWSIM bag解析
├── src/lidar_racing_rl/      # 環境、geometry、NPC、SAC、評価、model変換
├── tests/                    # unit / contract test
├── compose.yaml              # CPU向け専用コンテナ
├── compose.gpu.yaml          # NVIDIA GPU overlay
├── pyproject.toml
└── THIRD_PARTY_NOTICES.md
```

## F1TENTH Gym JAXとlockfileの準備

submodule登録は完了しています。現在のfork変更は未commitなので、まずsubmodule内の差分をreviewし、`Arata-Stu/f1tenth_gym_jax`へcommit/pushしてから、親リポジトリで新しいgitlinkを固定します。公式upstreamのpush URLは`DISABLED`です。浮動`main`を直接依存先にしてはいけません。transitiveなray marcher `jax-pf`も監査した完全SHAで固定しています。

submoduleの確認:

```bash
git submodule update --init --recursive
git -C aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax remote -v
git -C aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax rev-parse HEAD
```

初回setupだけはlockfile生成を許可するため、`uv sync`を非frozenで実行します。

```bash
# CPU。imageをbuildし、uv.lockを生成してCPU named volumeへ同期する
make lidar-rl-setup

# NVIDIA。CUDA extraを使い、GPU専用named volumeへ同期する
make lidar-rl-setup LIDAR_RL_GPU=1
```

生成された`uv.lock`とsubmodule SHAをCPU/GPU環境で確認してから管理対象にします。以後の`lidar-rl-test`、benchmark、train、evaluation、exportターゲットはすべて`uv run --frozen`を使い、lockfileとの不一致を失敗させます。

backend確認は専用コンテナ内で行います。

```bash
cd aichallenge/ml_workspace/lidar_racing_rl
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
docker compose run --rm --no-deps lidar-rl \
  uv run --frozen --extra f1tenth python scripts/check_backend.py
```

macOSではCPUコンテナによる静的・軽量確認だけを想定します。CUDA学習とAWSIM統合はUbuntu x86_64/NVIDIA環境で確認してください。

## 単独Pure Pursuit rollout

`run_single_rollout.py`は既定で`simulator.track.centerline`からPure Pursuit actionを毎step計算します。F1TENTH既定racelineはSpielbergの境界へ寄りすぎ、幅1.45 mのカートには安全な基準線ではないため使用しません。NPCの横offsetもコース幅と車体幅から得る余裕内か実行前に検証します。GT poseは許可された教師・NPC制御だけに使用し、環境wrapperが返すActor観測はLiDARだけです。教師を選んだのに車両が動かない場合や非有限値が出た場合も成功扱いにしません。

setup完了後、CPU専用コンテナで実行します。

```bash
cd aichallenge/ml_workspace/lidar_racing_rl
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
docker compose run --rm --no-deps lidar-rl \
  uv run --frozen --extra f1tenth python scripts/run_single_rollout.py \
  --config-name step1_single_vehicle \
  --action-source pure-pursuit \
  --steps 1000 \
  --output outputs/single_rollout.json
```

JAXを初期化せず設定だけを検証するには末尾へ`--dry-run`を追加します。固定actionのAPI smokeが必要な場合だけ`--action-source fixed`と`--ego-*` / `--npc-*`を指定します。

## 64環境benchmark

既定値はblueprintの`num_envs=64`、`num_agents=4`、`num_beams=360`、`steps=1000`です。JAXの環境・beam軸にPython逐次loopを置かず、compile時間、rollout時間、環境・車両step/s、peak memoryをJSONへ出します。Ego actionは固定zero、NPC 3台はエピソード単位で多様化したPure Pursuit・GT安全車間制御・制動イベント・制御遅延を使用し、Egoのauto-reset環境だけNPC状態も再初期化します。

```bash
# CPU
make lidar-rl-benchmark \
  LIDAR_RL_ARGS='--output outputs/benchmark.json'

# NVIDIA
make lidar-rl-benchmark LIDAR_RL_GPU=1 \
  LIDAR_RL_ARGS='--output outputs/benchmark-gpu.json'

# JAXを初期化しないconfig確認
make lidar-rl-benchmark LIDAR_RL_ARGS='--dry-run'
```

## Test・学習・評価・export

次のコマンドはlockfileと依存環境を用意した後の正規入口です。現在のworktreeでは未実行です。

```bash
# macOSでも実行できる、第三者依存なしのPython/TOML/JSON/XML・情報境界検査
make lidar-rl-static

make lidar-rl-test

make lidar-rl-train-step1 LIDAR_RL_ARGS='--dry-run'
make lidar-rl-train-step2 LIDAR_RL_ARGS='--dry-run'
make lidar-rl-eval LIDAR_RL_ARGS='--dry-run --episodes 10'
make lidar-rl-export LIDAR_RL_ARGS='--dry-run --output exported/'
```

依存環境を用意したUbuntuホストでは、まず10,000 transitionのStep 1 smokeを実施します。学習はrootとforkのcommit SHAを成果物へ記録するため、未commit差分がある状態ではfail closedします。

```bash
make lidar-rl-train-step1 \
  LIDAR_RL_GPU=1 \
  LIDAR_RL_ARGS='--max-transitions 10000 --output outputs/smoke-step1'

# Replayを再収集するwarm restart。bit-exactな継続ではない
make lidar-rl-train-step1 \
  LIDAR_RL_GPU=1 \
  LIDAR_RL_ARGS='--max-transitions 10000 --output outputs/smoke-step1-resume training.resume_from=outputs/smoke-step1/checkpoints'

make lidar-rl-eval \
  LIDAR_RL_GPU=1 \
  LIDAR_RL_ARGS='--config-name step1_single_vehicle --checkpoint outputs/smoke-step1/checkpoints --episodes 10 --output outputs/smoke-step1/evaluation.json'

make lidar-rl-export \
  LIDAR_RL_GPU=1 \
  LIDAR_RL_ARGS='--checkpoint outputs/smoke-step1/checkpoints --output exported/smoke-step1'
```

checkpointにはReplay Bufferと環境状態を含めません。再開時はActor/Critic/optimizer/temperatureと累積進捗を復元し、新しいReplayを教師方策で再warmupしてから更新を再開します。

smokeでNaNなし、Replay sample、Actor/Critic更新、checkpoint保存・warm restart、決定論評価、Flax/PyTorch parityを確認してから、`--max-transitions`を外してStep 1本学習へ進みます。Step 2のsourceには相対進行、passヒステリシス、安全接触、追従停滞の報酬と評価指標まで接続済みですが、実行はStep 1成立、4台rollout、NPC安全性をUbuntu上で確認してから行ってください。

export済みbundleはchecksumを検証してROS 2パッケージへ配置します。既存modelの置換は明示的に`LIDAR_RL_INSTALL_ARGS=--force`を指定します。

```bash
make lidar-rl-install-policy \
  LIDAR_RL_BUNDLE=aichallenge/ml_workspace/lidar_racing_rl/exported/smoke-step1
```

## AWSIM LiDAR計測

AWSIMへ接続する処理には専用JAXコンテナを使いません。既存AI Challenge dev imageを用意し、LiDARを有効にする専用scenarioを使います。一般の`make dev`はAWSIMのLiDARをoffにするため、この計測入口には使いません。

```bash
./docker_build.sh dev

run_id="$(date +%Y%m%d-%H%M%S)"
make simulator-lidar-rl LOG_DIR="/output/${run_id}" LIDAR_RL_VEHICLES=1
LOG_DIR="/output/${run_id}" ROS_DOMAIN_ID=1 docker compose up -d rosbag

# 必要な走行を終えたらbagを安全にcloseする
docker compose stop rosbag
```

保存したLaserScanは、ROS 2のbag APIが入った同じAI Challenge環境で解析します。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD="python3 scripts/analyze_awsim_bag.py /output/${run_id}/d1/rosbag2_all" \
docker compose run --rm --no-deps autoware-command
```

既定出力は`assets/calibration/lidar_statistics.json`と`.md`です。統計の解釈と追加optionは[calibration README](assets/calibration/README.md)を参照してください。終了時は`make down`を実行します。

## AWSIM 車両応答計測

操舵・加速度応答は、正規control command、実操舵status、velocity statusを同じall-topic bagへ
記録し、明示した実験区間だけから同定します。入力振幅や時間を推測した自動励起は行わないため、
安全な専用シナリオで操舵step、sine sweep、加速度step、coast、定常旋回を収録してください。

```bash
cp aichallenge/ml_workspace/lidar_racing_rl/assets/calibration/awsim_vehicle_experiments.template.json \
  output/awsim_vehicle_experiments.json

CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_vehicle_response.py \
  /output/<run-id>/d1/rosbag2_all \
  --experiments /output/awsim_vehicle_experiments.json' \
docker compose run --rm --no-deps autoware-command
```

テンプレートの`null`はbagで確認した区間・受入値へ置き換えます。欠topic、stamp異常、励起不足、
低い回帰品質は推測で補わず、理由付き`null`として
`assets/calibration/awsim_vehicle_model.yaml`へ保存します。7値が揃わない解析は終了code 3です。
詳細は[calibration README](assets/calibration/README.md)を参照してください。

## AWSIM ROS推論

開発時はLiDAR有効scenarioと`lidar_racing_controller`をまとめて既存AI Challenge Dockerで起動します。

```bash
# 初回またはcontroller変更後
./docker_build.sh dev
make autoware-build

# 1台、CPU LiDAR
make lidar-rl-awsim

# ROS graphの起動後、AWSIMへ制御許可を送る
make lidar-rl-request-control

# 4台、ROS_DOMAIN_ID 1..4
make lidar-rl-awsim4
make lidar-rl-request-control4

# 対応するUbuntu/NVIDIAホストでGPU LiDAR
LIDAR_MODE=gpu make lidar-rl-awsim
```

制御許可はAutowareコンテナとAWSIMのtopic discovery後に実行します。車両が動かない場合は、まず各domainのcontrollerログで「verified policy loaded」「fail-safe cleared」を確認し、model不在、LaserScan metadata不一致、有効beam不足、scan timeoutのいずれかを解消してください。停止は`make down`です。

sealed評価imageへcontrollerとexport済みmodelを組み込んだ後は、次を正規入口とします。`CONTROL_METHOD=lidar_racing`ならLiDARは未指定時にCPU、従来のMPC評価ではoffになります。

```bash
./create_submit_file.bash
./docker_build.sh eval --submit submit/aichallenge_submit.tar.gz
CONTROL_METHOD=lidar_racing make eval
```

現在は学習済みmodelがなく、開発・sealed評価ともend-to-end成功を確認していません。model／manifest未配置時はcontrollerの安全停止が継続するのが期待動作です。

## 設定と生成物

- `configs/train/step1_single_vehicle.yaml`: 単車両LiDAR-only学習
- `configs/train/step2_four_vehicle.yaml`: Ego 1台とPure Pursuit NPC 3台
- `configs/env/`: canonical scan、動的車両LiDAR、将来のdomain randomization契約
- `configs/vehicle/aichallenge_kart.yaml`: 車両寸法とAWSIM同定値
- `configs/npc/pure_pursuit.yaml`: NPC横・縦制御と多様化
- `configs/agent/sac.yaml`: LiDAR-only SAC契約
- `configs/deployment/awsim.yaml`: AWSIMトピック、前処理、フェイルセーフ

設定値に`null`が残る項目はAWSIM rosbagによる同定が必要です。推測値のまま本番設定として確定しないでください。

学習ログ、checkpoint、Replay Buffer、通常の`.pt` / `.msgpack`はGit管理外です。最終提出modelだけをmanifestとともにROS 2推論パッケージへ配置します。第三者コードの調査SHA、実際のsubmodule SHA、ライセンスを混同せず[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)へ記録します。
