# ADR: E2E LiDAR RLの実行Docker境界

- 状態: 採用
- 日付: 2026-08-27

## コンテキスト

E2E LiDAR RLには二種類の実行系がある。

1. AWSIM、ROS 2、Autoware、LaserScan、Ackermann制御へ接続する計測・推論と、将来のonline学習
2. F1TENTH Gym JAX上で完結するoffline学習、batch benchmark、unit test、設定検証、モデル変換

既存AI Challenge Dockerは前者の正規環境であり、ROS 2 Humble、Autoware、AWSIM、PyTorch、評価時のnetwork・device・launch境界を持つ。一方、後者はJAX、Flax、Optax、F1TENTH Gym JAXを使用し、AWSIMやROS graphを必要としない。

JAXとCUDAの組合せはホストのNVIDIA Driverとの互換性に左右される。JAX一式を既存AI Challenge imageへ追加すると、ROS推論に不要な依存がsealed評価imageへ入り、既存PyTorch環境との解決競合やimage肥大化を招く。逆に、AWSIMへ接続する処理を学習専用imageへ移すと、大会の正規ROS・sensor・評価境界を再構築することになる。

## 決定

1. 最上位の境界は「AWSIMへ接続するか」とする。
   - AWSIMへ接続する教師データ生成、LiDAR・車両応答計測、rosbag解析、ROS 2推論、sealed評価は既存AI Challenge Dockerで実行する。将来AWSIM online学習を追加する場合もこの環境を使う。
   - AWSIMへ接続しないF1TENTH/JAXのoffline学習、環境benchmark、unit test、設定検証、モデル変換は`lidar_racing_rl`専用Dockerで実行する。
2. 専用imageは`python:3.12-slim-bookworm`を基にし、既存Autoware imageを継承しない。ルート`requirements.txt`へJAXを追加しない。
3. Python依存の正本は`lidar_racing_rl/pyproject.toml`と`uv.lock`とする。
   - 初回`make lidar-rl-setup`だけは`uv sync`を非frozenで実行し、submodule固定後の`uv.lock`生成を許可する。
   - lock生成後のtest、benchmark、train、evaluation、exportは`uv run --frozen`を使い、定義とlockの不一致を失敗させる。
   - fork submoduleは登録済みだが変更commit SHAと`uv.lock`は未確定なので、setup完了と依存実行は未検証である。
4. リポジトリrootを`/workspace`へbind mountする。これによりソース変更を即時反映し、root repositoryとF1TENTH Gym JAX submoduleのcommit SHAを実験記録へ残す。
5. コンテナはホストの`HOST_UID` / `HOST_GID`で実行し、学習成果物をホストユーザー所有で作成する。
6. 仮想環境とuv cacheはnamed volumeへ分離する。macOS、Linux CPU、Linux CUDAの間でホスト上の`.venv`を共有しない。
7. CPU Composeを既定とし、NVIDIA GPUは`compose.gpu.yaml`の明示的なoverlayにする。`LIDAR_RL_GPU=1`のときだけCUDA extraとGPU deviceを有効にし、CPU/GPUの仮想環境volumeも分離する。
8. 専用Dockerの入口はルートMakefileの`lidar-rl-*`targetへ統一する。追加のCLI引数は`LIDAR_RL_ARGS`、初回sync引数は`LIDAR_RL_SYNC_ARGS`で渡す。
9. 専用DockerはROS_DOMAIN_ID、host network、X11、`privileged`、AWSIM deviceを要求せず、AWSIMへ接続しない。境界を越えるのは、既存環境で計測したcalibration artifactと、専用環境で生成したexport済みmodel・manifestに限定する。
10. AWSIM開発推論は`make lidar-rl-awsim` / `lidar-rl-awsim4`、sealed評価は`CONTROL_METHOD=lidar_racing make eval`を正規入口とする。一般の`make dev`はLiDARをoffにするため、この用途へ流用しない。
11. 現在の実装範囲にAWSIM online学習・fine-tuningは含まれない。Step 3はF1TENTHで学習したActorのexport、ROS 2推論、AWSIM評価までとする。

## 影響

### 利点

- JAX学習依存がAutoware/AWSIMおよび提出時のPyTorch runtimeへ流入しない。
- AWSIM計測・推論は大会のROS topic、domain、sensor、sealed評価境界をそのまま使える。
- CPU環境とNVIDIA環境の選択が明示的になり、backend情報を実験記録へ残せる。
- named volumeによりホストOSやCPU architectureの異なる仮想環境を誤用しない。
- Makefile経由でホストUID/GIDが渡るため、`outputs/`をroot所有にしない。
- 初回lock生成と、その後のfrozen実行がコマンド上で区別される。

### トレードオフ

- 初回はfork submoduleの固定・初期化、専用imageのbuild、`uv.lock`生成、named volumeへのsyncが必要になる。
- `uv.lock`を意図して変更する場合は、非frozenの保守手順で再生成し、CPU/GPU双方で確認する必要がある。
- GPU overlayを有効にしてもDriver互換性は保証されない。lockされたCUDA依存と`check_backend.py`の確認が別途必要である。
- AWSIMで得たrosbag/calibrationと、専用環境の学習artifactを明示的に受け渡す運用が必要になる。
- macOS arm64上のCPUコンテナは静的・軽量確認には使えるが、AWSIMとNVIDIA学習の正準環境ではない。正準環境はUbuntu x86_64/NVIDIAホストとする。

## 不採用案

### 既存`aichallenge-2025-dev`へJAX一式を追加する

ROS 2推論に不要な依存が全serviceへ入り、既存Python環境との競合とsealed評価imageの肥大化を招くため不採用とした。

### AWSIM計測・推論を専用JAX imageへ移す

大会が提供するROS 2、Autoware、AWSIM、network、device、評価imageの境界を複製することになり、正規環境との差分が増えるため不採用とした。

### ホストの`uv`を正規実行環境にする

macOSとLinux、CPUとCUDAで環境差が生まれ、専用DockerをF1TENTH/JAX側の正準環境とする方針に反するため不採用とした。

### `.venv`をリポジトリへbind mountする

OS・architecture固有wheelや絶対pathを異なる実行環境で再利用する危険があるため不採用とした。
