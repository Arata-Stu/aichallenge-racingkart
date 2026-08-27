# AI Challenge E2E LiDAR Racing RL プロジェクト定義書

## 0. 文書情報

| 項目              | 内容                                                                      |
| --------------- | ----------------------------------------------------------------------- |
| プロジェクト名         | AI Challenge E2E LiDAR Racing RL                                        |
| 開発リポジトリ         | `Arata-Stu/aichallenge-racingkart`                                      |
| ベースブランチ         | `e2e-dev`                                                               |
| 新規開発ブランチ        | `feat/e2e-lidar-sac-jax`                                                |
| 学習コード配置先        | `aichallenge/ml_workspace/lidar_racing_rl/`                             |
| AWSIM推論パッケージ配置先 | `aichallenge/workspace/src/aichallenge_submit/lidar_racing_controller/` |
| 学習アルゴリズム        | Soft Actor-Critic（SAC）                                                  |
| 学習フレームワーク       | JAX / Flax / Optax                                                      |
| 主観測             | 2D LiDARのみ                                                              |
| 車両構成            | 学習対象1台＋NPC 3台                                                           |
| 並列化             | 1環境4台を複数環境`vmap`                                                        |
| 最終用途            | 自動運転AIチャレンジ End to End部門 AWSIM走行                                        |

---

# 1. 背景

自動運転AIチャレンジのEnd to End部門では、Camera、LiDAR、Steer Angle、Wheel Odometry、Gear Statusが使用可能とされている。ただし、本プロジェクトでは競技ルールより厳しい条件として、**学習済み方策の入力を2D LiDARのみに限定する**。

End to End部門のSIM決勝は4台同時、6周のレースとして定義されている。一方、ルールページはWIPと明記されており、今後変更される可能性があるため、提出前に必ず再確認する。

AWSIMがPublishする2D LiDARは`/sensing/lidar/scan`の`LaserScan`で、現行仕様では1080点、最大検出距離30mである。制御は`/control/command/control_cmd`へ目標操舵角と目標加速度を送る。

AWSIMはGUI依存が強く、JAX環境のような大量の同期ステップ学習には適さない。そのため、本プロジェクトでは以下の三段階構成を採用する。

1. F1TENTH Gym JAXによる単車両学習
2. 4台環境による追従・追い抜き学習
3. AWSIMへのモデル転移とドメイン適応

---

# 2. プロジェクトの目的

## 2.1 最終目的

2D LiDARの時系列のみから、AI Challengeのレーシングカートに対する操舵角と加速度を出力し、以下の行動を実現する。

* 単独での高速周回
* 前方車両への安全な追従
* 追い抜けない区間での待機
* 左右の空き空間を利用した追い抜き
* 並走状態からの安全な復帰
* LiDARの欠損、壁抜け、観測遅延に対する頑健性
* AWSIM上での実時間推論

## 2.2 技術目標

* 1環境あたり4台を保持する。
* 4台のうち1台だけをSACで学習する。
* 残り3台はPure Pursuitを基本とする固定NPCとする。
* 1環境を`jax.vmap`して64環境以上を同時実行できる構造にする。
* 動的車両を2D LiDARに反映する。
* ActorとCriticの入力にGT位置、地図座標、Frenet座標を含めない。
* GT情報は報酬、教師方策、NPC制御、評価、初期配置に限って使用する。
* JAXで学習したActorをAWSIM用ROS 2ノードへ移植できるようにする。
* 学習環境とAWSIM推論環境で同一のLiDAR前処理を使用する。

---

# 3. 対象外

初期実装では以下を対象外とする。

* 4台すべてを同時に学習するMulti-Agent SAC
* V2X、RSU LiDAR、GNSS、IMU、自己位置をActorへ入力する構成
* カメラとLiDARの融合
* AWSIM上での大量オンラインRL
* Transformerや大規模World Model
* 実車への直接転移
* 他車両のGT位置を推論時に利用する構成
* Virtual LiDARをActorへ追加観測として与える構成
* 既存の`reinforcement_learning`実装の置換または削除

既存の`aichallenge/ml_workspace/reinforcement_learning/`は、AWSIMと直接通信しながらStable-Baselines3のSACを実行する既存ベースラインとして残す。

---

# 4. Git・ブランチ運用

## 4.1 親リポジトリ

開発対象は以下とする。

```bash
https://github.com/Arata-Stu/aichallenge-racingkart.git
```

このリポジトリは公式`AutomotiveAIChallenge/aichallenge-racingkart`のforkである。既存ブランチとして`main`、`dev`、`e2e-dev`が存在する。

## 4.2 ブランチ作成

Code Agentは次の順番で作業ブランチを作成する。

```bash
git clone https://github.com/Arata-Stu/aichallenge-racingkart.git
cd aichallenge-racingkart

git remote add upstream \
  https://github.com/AutomotiveAIChallenge/aichallenge-racingkart.git

git fetch --all --prune

git switch e2e-dev
git pull --ff-only origin e2e-dev

git status --short
git switch -c feat/e2e-lidar-sac-jax
```

`git status --short`に出力がある場合、Code Agentは既存変更を破棄せず、作業を中断して状態を報告する。

## 4.3 マージ先

最終的なPull Requestのbaseは`e2e-dev`とする。

```text
feat/e2e-lidar-sac-jax
           ↓
        e2e-dev
```

`main`や公式`upstream/dev`へ直接マージしない。

## 4.4 コミット方針

Conventional Commitsに準じる。

```text
chore: initialize lidar racing rl workspace
feat: add vectorized four-vehicle environment
feat: add dynamic vehicle lidar scan
feat: add jax pure pursuit opponent controller
feat: implement lidar-only sac
feat: add awsim policy export
feat: add ros2 lidar racing controller
test: add dynamic lidar geometry tests
docs: add training and deployment guide
```

一つのコミットに環境実装、SAC、ROS 2推論を混在させない。

---

# 5. 外部GitHubリポジトリ管理

## 5.1 基本原則

外部コードは以下の3種類に分類する。

| 分類                     | 管理方法                       | 例                       |
| ---------------------- | -------------------------- | ----------------------- |
| 通常のPythonライブラリ         | `pyproject.toml`と`uv.lock` | JAX、Flax、Optax、Flashbax |
| 実際にimportし、独自変更するリポジトリ | fork＋Git submodule         | F1TENTH Gym JAX         |
| 実装を読むだけの参考リポジトリ        | `.repos`ファイルに記録            | Stoix、End2Race等         |

外部GitHubコードをコピー＆ペーストして出典を失わせてはならない。

## 5.2 F1TENTH Gym JAX

F1TENTH Gym JAXは、JIT可能なマルチエージェント環境として`reset`と`step`を提供している。公式の導入方法はPython 3.11〜3.13の独立環境と`uv`を使用する方法である。

一方、現行実装のLiDAR生成は静的地図に対するray marchingとガウスノイズのみであり、車両の矩形は衝突判定には使われるがLiDAR生成には使用されていない。動的車両をスキャンへ追加するには、`_scan()`周辺の変更が必要になる可能性が高い。

したがって、F1TENTH Gym JAXは最初からforkを作成し、forkをsubmoduleとして使用する。

### fork

```text
upstream:
    f1tenth/f1tenth_gym_jax

fork:
    Arata-Stu/f1tenth_gym_jax

fork側作業ブランチ:
    aichallenge/dynamic-lidar
```

forkが未作成の場合は、GitHub上または認証済みGitHub CLIで作成する。fork作成に失敗した状態でupstreamへ変更を加えてはならない。

### submodule配置先

```text
aichallenge/ml_workspace/lidar_racing_rl/
└── repos/
    └── f1tenth_gym_jax/
```

fork作成後の登録例：

```bash
git submodule add \
  https://github.com/Arata-Stu/f1tenth_gym_jax.git \
  aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax

cd aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax

git remote add upstream \
  https://github.com/f1tenth/f1tenth_gym_jax.git

git fetch upstream
git switch -c aichallenge/dynamic-lidar origin/main
```

親リポジトリではsubmoduleの特定コミットを固定する。常に最新`main`へ追従する設定にはしない。

クローン時は以下を使用する。

```bash
git clone --recurse-submodules \
  https://github.com/Arata-Stu/aichallenge-racingkart.git
```

既存クローンでは以下を使用する。

```bash
git submodule update --init --recursive
```

## 5.3 forkに加える変更の範囲

fork側には、他プロジェクトでも再利用可能な汎用変更だけを実装する。

許可する変更：

* 動的OBBをLiDARへ反映する機能
* 動的障害物スキャンを差し込むためのhook
* scan corruption hook
* 初期車両配置を外部から指定するhook
* `terminated`と`truncated`の分離
* 必要なベンチマーク・単体テスト
* LiDAR-only観測を返すための汎用設定

fork側へ置かないもの：

* AI Challenge固有の報酬
* AI Challenge固有のNPC速度プロファイル
* SAC学習ループ
* ROS 2ノード
* AWSIMのトピック名
* CCTB専用のハードコード

これらは親リポジトリ側に実装する。

## 5.4 参考リポジトリ

以下のようなリポジトリは、直接importしない限りsubmoduleにしない。

```text
Stoix
End2Race
f1tenth_development_gym
```

以下のファイルへ、用途、URL、参照コミット、ライセンスを記録する。

```text
aichallenge/ml_workspace/lidar_racing_rl/repos/reference.repos
aichallenge/ml_workspace/lidar_racing_rl/repos/README.md
```

`reference.repos`はvcstool互換形式とする。

```yaml
repositories:
  stoix:
    type: git
    url: https://github.com/EdanToledo/Stoix.git
    version: <固定コミットSHA>

  end2race:
    type: git
    url: https://github.com/michigan-traffic-lab/End2Race.git
    version: <固定コミットSHA>

  f1tenth_development_gym:
    type: git
    url: https://github.com/F1Tenth-INI/f1tenth_development_gym.git
    version: <固定コミットSHA>
```

`main`、`master`、`latest`などの浮動参照は禁止する。

## 5.5 ライセンス

親のAI ChallengeリポジトリはApache License 2.0、F1TENTH Gym JAXはMIT Licenseである。F1TENTH Gym JAXのMITライセンスは、再配布時に著作権表示と許諾表示を含めることを要求している。

以下を作成する。

```text
aichallenge/ml_workspace/lidar_racing_rl/THIRD_PARTY_NOTICES.md
```

このファイルには最低限、以下を記録する。

* リポジトリ名
* upstream URL
* fork URL
* 固定コミットSHA
* ライセンス
* 使用箇所
* 独自変更の概要

---

# 6. Python環境

## 6.1 既存環境との分離

ルート`requirements.txt`にはPyTorchとStable-Baselines3が含まれているが、現時点ではJAXは含まれていない。

JAX学習環境は既存のAutoware/AWSIM用Docker環境へ直接混在させず、次のサブプロジェクト内で独立管理する。

```text
aichallenge/ml_workspace/lidar_racing_rl/pyproject.toml
aichallenge/ml_workspace/lidar_racing_rl/uv.lock
```

初期段階ではルート`requirements.txt`を変更しない。

## 6.2 使用候補ライブラリ

```text
jax
flax
optax
distrax
flashbax
orbax-checkpoint
hydra-core
omegaconf
numpy
scipy
pandas
matplotlib
tensorboard
pytest
ruff
```

バージョンはF1TENTH Gym JAXの`pyproject.toml`および`uv.lock`と整合させる。

## 6.3 セットアップ

```bash
cd aichallenge/ml_workspace/lidar_racing_rl
uv sync
uv run python scripts/check_backend.py
```

`check_backend.py`は以下を表示する。

* Pythonバージョン
* JAXバージョン
* JAX backend
* 使用可能デバイス
* GPU名
* XLA設定
* submodule commit SHA

CUDA版JAXの導入は、実行ホストのNVIDIA DriverとCUDA互換性を確認したうえで固定する。特定CUDAバージョンを無条件にルート環境へ追加しない。

---

# 7. ディレクトリ構成

```text
aichallenge-racingkart/
├── .gitmodules
├── docs/
│   └── e2e_lidar_racing_rl_definition.md
│
├── aichallenge/
│   ├── ml_workspace/
│   │   ├── reinforcement_learning/       # 既存。変更しない
│   │   │
│   │   └── lidar_racing_rl/
│   │       ├── README.md
│   │       ├── pyproject.toml
│   │       ├── uv.lock
│   │       ├── THIRD_PARTY_NOTICES.md
│   │       ├── .gitignore
│   │       │
│   │       ├── configs/
│   │       │   ├── train/
│   │       │   │   ├── step1_single_vehicle.yaml
│   │       │   │   └── step2_four_vehicle.yaml
│   │       │   ├── env/
│   │       │   │   ├── base.yaml
│   │       │   │   ├── dynamic_lidar.yaml
│   │       │   │   └── domain_randomization.yaml
│   │       │   ├── vehicle/
│   │       │   │   └── aichallenge_kart.yaml
│   │       │   ├── npc/
│   │       │   │   └── pure_pursuit.yaml
│   │       │   ├── agent/
│   │       │   │   └── sac.yaml
│   │       │   └── deployment/
│   │       │       └── awsim.yaml
│   │       │
│   │       ├── assets/
│   │       │   ├── maps/
│   │       │   ├── waypoints/
│   │       │   └── calibration/
│   │       │
│   │       ├── src/
│   │       │   └── lidar_racing_rl/
│   │       │       ├── envs/
│   │       │       │   ├── make_env.py
│   │       │       │   ├── vector_env.py
│   │       │       │   ├── observation.py
│   │       │       │   ├── reward.py
│   │       │       │   ├── termination.py
│   │       │       │   ├── reset_sampler.py
│   │       │       │   ├── domain_randomization.py
│   │       │       │   └── lidar_corruption.py
│   │       │       │
│   │       │       ├── geometry/
│   │       │       │   ├── ray_obb.py
│   │       │       │   └── dynamic_scan.py
│   │       │       │
│   │       │       ├── npc/
│   │       │       │   ├── pure_pursuit.py
│   │       │       │   ├── longitudinal_control.py
│   │       │       │   └── opponent_pool.py
│   │       │       │
│   │       │       ├── models/
│   │       │       │   ├── encoder_flax.py
│   │       │       │   ├── actor_flax.py
│   │       │       │   ├── critic_flax.py
│   │       │       │   ├── actor_torch.py
│   │       │       │   └── parameter_conversion.py
│   │       │       │
│   │       │       ├── sac/
│   │       │       │   ├── learner.py
│   │       │       │   ├── losses.py
│   │       │       │   ├── replay.py
│   │       │       │   ├── collector.py
│   │       │       │   └── train_state.py
│   │       │       │
│   │       │       ├── evaluation/
│   │       │       │   ├── metrics.py
│   │       │       │   ├── evaluator.py
│   │       │       │   └── scenarios.py
│   │       │       │
│   │       │       ├── export/
│   │       │       │   ├── export_policy.py
│   │       │       │   └── manifest.py
│   │       │       │
│   │       │       └── awsim/
│   │       │           ├── scan_statistics.py
│   │       │           ├── dynamics_identification.py
│   │       │           └── compare_domains.py
│   │       │
│   │       ├── scripts/
│   │       │   ├── check_backend.py
│   │       │   ├── train.py
│   │       │   ├── evaluate.py
│   │       │   ├── benchmark_env.py
│   │       │   ├── export_policy.py
│   │       │   └── analyze_awsim_bag.py
│   │       │
│   │       ├── tests/
│   │       │   ├── test_ray_obb.py
│   │       │   ├── test_dynamic_scan.py
│   │       │   ├── test_vector_env.py
│   │       │   ├── test_observation_boundary.py
│   │       │   ├── test_reward.py
│   │       │   ├── test_sac_smoke.py
│   │       │   └── test_export_parity.py
│   │       │
│   │       ├── repos/
│   │       │   ├── README.md
│   │       │   ├── reference.repos
│   │       │   └── f1tenth_gym_jax/      # Git submodule
│   │       │
│   │       └── outputs/                   # Git管理外
│   │
│   └── workspace/
│       └── src/
│           └── aichallenge_submit/
│               └── lidar_racing_controller/
│                   ├── package.xml
│                   ├── setup.py
│                   ├── setup.cfg
│                   ├── resource/
│                   ├── launch/
│                   ├── config/
│                   ├── models/
│                   ├── lidar_racing_controller/
│                   │   ├── node.py
│                   │   ├── preprocessing.py
│                   │   ├── policy.py
│                   │   └── safety.py
│                   └── test/
│
└── output/                              # Git管理外
```

大会リポジトリでは、学習用コードは`ml_workspace`、参加者が提出するROS 2パッケージは`aichallenge_submit`へ配置する。提出スクリプトは`aichallenge_submit`を圧縮する構造である。

---

# 8. 情報利用境界

## 8.1 Actor・Criticへ入力可能な情報

ActorとCriticに入力してよいのは、次の情報だけとする。

```text
現在および過去の2D LiDAR ranges
LiDARから算出したvalidity mask
LiDARから算出した局所的なフィルタ結果
```

LiDAR履歴は外界センサの時間系列なので使用可能とする。

## 8.2 Actor・Criticへの入力禁止情報

以下はActorとCriticへ入力してはならない。

```text
自車GT位置
自車GT姿勢
相手車両GT位置
相手車両GT速度
Frenet s / ey / epsi
地図上の自己位置
Waypoint index
Lap count
順位
V2X
GNSS
IMU
Wheel Odometry
Steer Angle
Gear Status
前回行動
教師方策の内部状態
```

公式ルール上はSteer Angle、Wheel Odometry、Gear Statusも使用可能だが、本プロジェクトではLiDAR-onlyという独自制約を優先する。

## 8.3 GT情報を使用可能な場所

GT情報は次の処理に限って使用可能とする。

* 報酬計算
* collision・off-track判定
* 追い抜き成立判定
* 初期配置
* NPC Pure Pursuit
* NPC車間制御
* 教師方策
* 評価指標
* AWSIMとのsystem identification
* シミュレータ内部でのLiDAR ray casting

動的車両LiDARの生成に相手GT位置を用いることは許可する。これはActorへの追加情報ではなく、シミュレータがセンサ観測を生成するために必要な処理である。

---

# 9. 全体アーキテクチャ

```text
                  学習時のみ使用
 ┌─────────────────────────────────────────────┐
 │ Map / GT pose / GT opponent state           │
 │             │                               │
 │             ├── Reward                      │
 │             ├── NPC Pure Pursuit            │
 │             ├── Reset sampler               │
 │             └── Optional teacher            │
 └─────────────────────────────────────────────┘

     F1TENTH Gym JAX: 1 environment = 4 vehicles
 ┌─────────────────────────────────────────────┐
 │ agent_0 : LiDAR-only SAC                    │
 │ agent_1 : Pure Pursuit + longitudinal ctrl  │
 │ agent_2 : Pure Pursuit + longitudinal ctrl  │
 │ agent_3 : Pure Pursuit + longitudinal ctrl  │
 │                                             │
 │ Static map scan + Dynamic vehicle OBB scan  │
 └─────────────────────────────────────────────┘
                     │
                     │ jax.vmap
                     ▼
        64～128 environments in parallel
                     │
                     ▼
              Replay Buffer
                     │
                     ▼
             JAX SAC Learner
                     │
                     ▼
           Flax actor parameters
                     │
          parameter conversion
                     ▼
             PyTorch actor
                     │
                     ▼
     ROS 2 lidar_racing_controller
                     │
          /sensing/lidar/scan
                     │
                     ▼
          /control/command/control_cmd
```

---

# 10. JAX環境仕様

## 10.1 配列形状

デフォルト構成を以下とする。

```text
num_envs      = 64
num_agents    = 4
num_beams     = 360
frame_stack   = 4
scan_channels = 2
action_dim    = 2
```

代表的な配列形状：

```text
vehicle_state:
    [num_envs, 4, state_dim]

raw_scan:
    [num_envs, 4, 360]

scan_history:
    [num_envs, 4, 4, 2, 360]

all_actions:
    [num_envs, 4, 2]

ego_observation:
    [num_envs, 4, 2, 360]

ego_action:
    [num_envs, 2]

ego_reward:
    [num_envs]

dynamic_scan_intermediate:
    [num_envs, observer_vehicle, target_vehicle, beam]
```

学習用transitionは各環境の`agent_0`のみを保存する。

1回のvectorized stepで得られるSAC transition数は`num_envs`個とする。

## 10.2 Pythonループ禁止範囲

以下の軸をPythonの逐次ループで処理してはならない。

* 並列環境
* 観測車両
* 相手車両
* LiDAR beam
* SAC update batch

`jax.vmap`、broadcast、`jax.lax.scan`を使用する。

設定読込や評価レポート作成など、JIT対象外の処理にはPythonループを使用してよい。

---

# 11. 動的車両LiDAR

## 11.1 基本式

各LiDAR beamについて、静的地図と動的車両までの距離の最小値を取る。

$$
r_{b,e,k}
=
\min
\left(
r^{map}_{b,e,k},
\min_{j\ne e}r^{vehicle}_{b,e,j,k}
\right)
$$

ここで、

* \(b\)：並列環境
* \(e\)：観測車両
* \(j\)：対象車両
* \(k\)：LiDAR beam

とする。

## 11.2 車両形状

各車両を向きを持つ長方形、すなわちOriented Bounding Boxとして扱う。

Rayと相手車両OBBの交差は以下の手順で求める。

1. beam originと方向を対象車両座標系へ変換
2. 対象車両を軸平行矩形として扱う
3. slab methodで交差距離を求める
4. 負距離と非交差を`max_range`へ置換
5. 自車自身をmask
6. 全対象車両の最小値を取得
7. 静的map scanとの最小値を取得

## 11.3 必須テスト

以下のケースを単体テストする。

* 正面の車両が正しい距離に映る
* 90度回転した車両が正しく映る
* 壁より手前の車両が映る
* 壁の後ろの車両は映らない
* 2台が重なる場合、手前だけが映る
* 自車自身がスキャンへ映らない
* 車両がFOV外の場合は影響しない
* 接線方向のbeamでNaNが発生しない
* `vmap`結果と単一環境結果が一致する

---

# 12. LiDAR前処理

## 12.1 Canonical Scan

学習モデルへ入力する標準形式を以下とする。

```text
beam count:
    360

range:
    0.0 ～ 1.0へ正規化

validity:
    valid = 1
    invalid = 0
```

AWSIMの1080点スキャンは3点ごとに集約して360点へ変換する。

初期デフォルトは、各グループ内の有効距離の最小値とする。

```text
AWSIM 1080 beams
        ↓
3-beam minimum pooling
        ↓
360 beams
```

壁を超えて遠方まで抜けるbeamが含まれても、隣接beamに正しい壁距離があれば近い値を残しやすい。

## 12.2 無効値処理

以下を無効値として扱う。

* `NaN`
* `Inf`
* `range < range_min`
* `range > range_max`
* センサ定義上の異常値

無効rangeは`range_max`へ置換する。ただし、必ず別チャンネルのvalidity maskを0にする。

```text
channel 0:
    normalized range

channel 1:
    validity mask
```

「最大距離」と「未観測」を同じ意味にしない。

## 12.3 壁抜け対策

AWSIMで発生する可能性がある、壁より遠い有効値を模擬する。

```text
far_leak:
    clean_rangeより遠いランダム距離へ置換

single_beam_dropout:
    単一beamをInf化

sector_dropout:
    連続する複数beamをInf化

frame_hold:
    過去フレームを再利用

frame_delay:
    観測を数step遅延

gaussian_noise:
    距離依存ノイズ

angle_bias:
    beam角度の微小ずれ
```

ノイズ値は固定の推測値ではなく、AWSIM rosbagの統計から設定する。

## 12.4 時系列

初期モデルは4フレームstackとする。

```text
scan t-3
scan t-2
scan t-1
scan t
```

各フレームにrangeとvalidityの2チャンネルを持つため、1D CNNの入力チャンネル数は8となる。

初期段階ではRecurrent SACを実装せず、frame stack SACを使用する。

GRU/LSTMはStep 2が成立した後の拡張とする。

---

# 13. 車両モデル

AI Challengeの現行シミュレータ仕様では、車両は全長2.0m、全幅1.45m、ホイールベース1.087m、最大加速度3.2m/s²とされている。これらを初期値として設定する。

```yaml
vehicle:
  length: 2.0
  width: 1.45
  wheelbase: 1.087
  max_acceleration: 3.2
```

ただし、仕様値だけでAWSIMの実効挙動を再現できるとは限らないため、以下はAWSIMの走行ログから同定する。

* 操舵ゲイン
* 操舵応答時定数
* 操舵遅延
* 加速度ゲイン
* 加速度遅延
* 速度抵抗
* 速度ごとの旋回応答
* 実効摩擦係数

環境は一つの固定値だけでなく、同定誤差を含む範囲からエピソード単位でパラメータをサンプリングできるようにする。

---

# 14. NPC仕様

## 14.1 基本構成

NPC 3台は次の組合せとする。

```text
横方向:
    Pure Pursuit

縦方向:
    Waypoint target speed
    +
    GTを利用した安全車間制御
```

Pure PursuitはJAX関数として実装し、全環境・全NPCを`vmap`する。

## 14.2 NPCの多様化

3台を完全に同じパラメータで走らせてはならない。

エピソード開始時に以下をランダム化する。

* Waypoint line
* lateral offset
* target speed multiplier
* lookahead distance
* steering gain
* control delay
* acceleration gain
* braking event
* safe following distance

例：

```yaml
npc:
  speed_multiplier:
    min: 0.65
    max: 1.05

  lateral_offset:
    min: -0.5
    max: 0.5

  lookahead:
    min: 1.5
    max: 4.0
```

## 14.3 NPC車間制御

通常のPure Pursuitは前方車両を無視するため、NPC同士の不自然な追突を避ける縦方向制御を追加する。

$$
v_{target}
=
\min
\left(
v_{waypoint},
v_{lead}+k_d(d-d_{safe})
\right)
$$

NPCの前方車両判定にはGT情報を使用してよい。

NPCは学習対象ではないため、そのtransitionをReplay Bufferへ追加しない。

---

# 15. SAC仕様

## 15.1 構成

標準的なcontinuous-control SACを実装する。

* Gaussian Actor
* Tanh squashing
* Twin Q Critic
* Target Q Network
* Automatic entropy tuning
* Polyak averaging
* Off-policy replay
* Gradient clipping
* NaN検出
* Checkpoint保存・再開

## 15.2 Actor

入力：

```text
[batch, frame_stack × channels, beams]
=
[batch, 8, 360]
```

初期ネットワーク：

```text
Conv1D
  ↓
Activation
  ↓
Conv1D
  ↓
Activation
  ↓
Conv1D
  ↓
Flatten
  ↓
Dense 256
  ↓
Actor mean head     [2]
Actor log_std head  [2]
```

`log_std`には上下限を設定する。

推論時は確率サンプルではなく平均行動を使用する。

## 15.3 Critic

CriticもLiDAR観測と行動だけを入力とする。

```text
LiDAR Encoder
      │
      ├── encoded feature
      │
action┘
      ↓
Twin Q MLP
```

CriticへGT位置やFrenet進行度を与えてはならない。

## 15.4 行動

Actorは正規化された2次元行動を出力する。

```text
action[0]:
    normalized steering angle

action[1]:
    normalized acceleration
```

環境内部で実値へ変換する。

$$
\delta
=
a_0\delta_{max}
$$

$$
a
=
a_{min}
+
\frac{a_1+1}{2}
(a_{max}-a_{min})
$$

AWSIMへは`AckermannControlCommand`として目標操舵角と目標加速度をPublishする。

## 15.5 Replay Buffer

Flashbaxまたは同等のJAX対応ring bufferを使用する。

保存対象：

```text
observation
action
reward
terminated
truncated
next_observation
```

保存しないもの：

```text
GT pose
GT opponent state
Frenet state
map coordinates
NPC transition
```

LiDARはメモリ削減のため、Replay Buffer内では`float16`または`uint16`表現を許可する。学習時に`float32`へ変換する。

## 15.6 終了処理

以下を`terminated`とする。

* Ego collision
* Ego off-track
* レース完了
* 回復不能状態

以下を`truncated`とする。

* 最大step
* 評価時間上限

時間上限による終了では、SAC targetのbootstrapを残す。

4台すべての終了を待たず、Egoが終了した時点でその並列環境をresetする。

---

# 16. 報酬設計

## 16.1 共通報酬

$$
\begin{aligned}
r_t={}&
w_p\Delta p_{ego}
+w_{rel}\Delta p_{relative}
+w_{pass}I_{pass}
\\
&-w_cI_{collision}
-w_oI_{offtrack}
-w_j\|u_t-u_{t-1}\|^2
-w_rI_{reverse}
\end{aligned}
$$

各項目：

```text
progress:
    自車のコース進行度

relative_progress:
    前方車両に対する相対進行度

pass:
    追い抜き成立イベント

collision:
    車両または壁との接触

offtrack:
    コース外

smoothness:
    行動の急変

reverse:
    後退
```

## 16.2 追い抜き判定

追い抜きは単純な順位入替ではなく、ヒステリシスを持つイベントとして判定する。

```text
相手より後方
    ↓
並走
    ↓
相手より一定距離前方
    ↓
一定時間維持
    ↓
pass成立
```

同一相手に対して短時間に複数回報酬を付与しない。

## 16.3 中心線ペナルティ

強い中心線偏差ペナルティは禁止する。

中心線追従を強くしすぎると、追い抜きに必要なライン変更が抑制される。

必要な場合は、コース境界への接近ペナルティまたは走行可能領域外ペナルティを使う。

## 16.4 Step別報酬

### Step 1

```text
progress
collision
offtrack
smoothness
reverse
```

`relative_progress`と`pass`は無効。

### Step 2

Step 1に以下を追加する。

```text
relative_progress
pass
unsafe contact
stalled behind vehicle
```

---

# 17. 学習段階

## Step 1：単車両LiDAR-only SAC

環境：

```text
agent_0:
    SAC

他車:
    なし
```

目的：

* LiDAR-onlyで周回できる
* コースアウトしない
* 操舵と加速度が発散しない
* LiDAR欠損に耐える
* 並列SAC実装を検証する

初期状態では安全な教師方策またはPure Pursuitへaction noiseを加え、Replay Bufferのwarmupに利用してよい。

教師方策はGTを使用してよいが、教師の入力や出力を推論時のActorへ追加してはならない。

## Step 1.5：静的障害物

任意の中間段階として、停止車両相当の矩形障害物をランダム配置する。

目的：

* 壁とは異なる小さな凸形状を認識する
* 静止障害物を避ける
* 動的相手を導入する前にRay–OBBを検証する

## Step 2：4台環境

環境：

```text
agent_0:
    SAC

agent_1:
    Pure Pursuit NPC

agent_2:
    Pure Pursuit NPC

agent_3:
    Pure Pursuit NPC
```

目的：

* 追従
* 衝突回避
* 左右からの追い抜き
* 並走
* 複数車両への対応

カリキュラム：

```text
2-A:
    遅いNPC 1台

2-B:
    速度の異なるNPC 3台

2-C:
    複数走行ライン

2-D:
    制御遅延・減速イベント

2-E:
    過去SAC checkpointをOpponent Poolへ追加
```

Opponent Poolを導入しても、学習対象は常に現在のEgo 1台だけとする。

## Step 3：AWSIM転移

目的：

* LiDAR前処理の一致
* 車両応答の一致
* センサ欠損への適応
* ROS 2推論
* 実時間制御

AWSIMは大量学習環境ではなく、次の用途へ限定する。

* LiDAR統計収集
* 車両system identification
* 固定シナリオ評価
* failure mining
* 推論遅延測定
* 最終モデル検証

---

# 18. AWSIM計測

## 18.1 LiDAR計測

rosbagから以下を集計する。

* `len(ranges)`
* `angle_min`
* `angle_max`
* `angle_increment`
* `range_min`
* `range_max`
* Publish周期
* 有効点率
* `NaN`率
* `Inf`率
* 連続欠損幅
* 距離別欠損率
* 壁抜け候補の頻度
* フレーム保持・遅延
* 角度ごとの欠損偏り

解析結果を以下へ保存する。

```text
assets/calibration/lidar_statistics.json
assets/calibration/lidar_statistics.md
```

## 18.2 車両応答計測

次の入力をAWSIMへ与える。

* 操舵step入力
* 操舵sine sweep
* 加速度step入力
* 加速後のcoast
* 一定速度旋回

推定結果：

```text
steering_gain
steering_time_constant
steering_delay
acceleration_gain
acceleration_delay
effective_wheelbase
velocity_drag
```

結果を以下へ保存する。

```text
assets/calibration/awsim_vehicle_model.yaml
```

---

# 19. JAXからAWSIMへのモデル移植

## 19.1 ランタイム方針

学習はFlax、AWSIM推論はPyTorchを使用する。

理由：

* AI Challengeの既存環境にはPyTorchが導入済み
* ROS 2側へJAX一式を追加する必要がない
* 提出物の依存関係を小さく保てる

## 19.2 Actor二重実装

以下の二つを同一アーキテクチャで実装する。

```text
encoder_flax.py
actor_flax.py

actor_torch.py
```

JAXの重みをPyTorchへ変換するスクリプトを実装する。

```bash
uv run python scripts/export_policy.py \
  --checkpoint outputs/<run>/checkpoint \
  --output exported/
```

出力：

```text
exported/
├── policy_flax.msgpack
├── policy_torch.pt
├── policy_manifest.json
└── config_snapshot.yaml
```

## 19.3 変換精度

同一のランダム入力に対し、Flax ActorとPyTorch Actorの決定論的出力を比較する。

合格条件：

```text
max absolute error <= 1e-5
```

## 19.4 Manifest

`policy_manifest.json`へ以下を保存する。

* architecture version
* beam count
* frame stack数
* range normalization
* validity定義
* action scaling
* training config hash
* root repository commit SHA
* F1TENTH Gym JAX submodule commit SHA
* model checksum
* export timestamp

---

# 20. ROS 2推論パッケージ

## 20.1 入力・出力

Subscribe：

```text
/sensing/lidar/scan
    sensor_msgs/msg/LaserScan
```

Publish：

```text
/control/command/control_cmd
    autoware_auto_control_msgs/msg/AckermannControlCommand
```

Actor用の入力として、それ以外のセンサ・状態トピックをSubscribeしない。

## 20.2 ノード処理

```text
LaserScan callback
      ↓
validate metadata
      ↓
1080 → 360 pooling
      ↓
normalize ranges
      ↓
generate validity mask
      ↓
update four-frame buffer
      ↓
PyTorch Actor inference
      ↓
action scaling
      ↓
rate limit
      ↓
AckermannControlCommand publish
```

## 20.3 フェイルセーフ

以下の場合は加速指令を停止または制動側へ移す。

* LiDARが一定時間届かない
* 有効beam率が閾値未満
* Actor出力にNaNまたはInf
* 推論例外
* モデルファイル不整合
* Manifestと設定のbeam数が一致しない

フェイルセーフ判定に使用する受信時刻やノード状態は、Actor入力には含めない。

## 20.4 推論周期

初期値：

```text
LiDAR callback:
    センサPublish周期

Control timer:
    20 Hz
```

実際のLiDAR周期を計測後、設定ファイルで変更可能にする。

---

# 21. 設定管理

HydraまたはOmegaConfを使用し、コード内へ学習条件をハードコードしない。

全runで以下を保存する。

```text
resolved_config.yaml
git_status.txt
git_diff.patch
root_commit.txt
submodule_commits.txt
environment.txt
metrics.jsonl
tensorboard/
checkpoints/
```

再現性のために乱数seedを以下へ分離する。

```yaml
seed:
  master: 0
  environment: 1
  reset: 2
  action: 3
  replay: 4
  evaluation: 5
```

---

# 22. 評価指標

## 単独走行

* 完走率
* 平均進行距離
* Lap time
* Collision rate
* Off-track rate
* 平均速度
* 操舵変化量
* 加速度変化量

## 他車あり

* Overtake success rate
* Time to overtake
* Collision rate with opponent
* Collision rate with wall
* Follow duration
* Minimum opponent distance
* Race completion rate
* Final rank
* Pass後の再接触率
* 抜けない区間での待機成功率

## システム

* Environment steps/sec
* SAC updates/sec
* GPU memory
* Replay Buffer memory
* JIT compile time
* ROS inference latency p50
* ROS inference latency p95
* Control publish rate
* LiDAR dropped-frame rate

---

# 23. テスト要件

## 23.1 Unit Test

```text
Ray–OBB geometry
Dynamic scan occlusion
LiDAR normalization
Invalid value handling
Wall-leak corruption
Frame stack reset
Progress wraparound
Pass hysteresis
Ego-only termination
Action scaling
Flax–PyTorch conversion
```

## 23.2 Integration Test

```text
1 environment × 4 vehicles
8 environments × 4 vehicles
64 environments × 4 vehicles
```

確認項目：

* 形状が固定されている
* JIT再コンパイルが反復中に発生しない
* 全車両がLiDARへ映る
* 自車自身が映らない
* NPCがwaypointを追従する
* Ego終了時にその環境だけresetされる
* 他の並列環境は継続する
* Actor観測にGT状態が含まれない

## 23.3 SAC Smoke Test

最低限、以下を満たす。

```text
10,000 environment transitions以上動作
NaNなし
checkpoint保存成功
checkpoint再開成功
deterministic evaluation成功
Replay Buffer sample成功
Actor・Critic update成功
```

学習性能そのものは短時間smoke testの合格条件にしない。

## 23.4 性能ベンチマーク

以下の条件でベンチマークする。

```text
num_envs   = 64
num_agents = 4
num_beams  = 360
steps      = 1000
```

出力：

```text
compile_seconds
rollout_seconds
environment_steps_per_second
vehicle_steps_per_second
peak_memory
```

特定GPUに依存する絶対速度は合否条件にせず、結果を記録する。Pythonの環境・beam逐次ループがないことを合格条件とする。

---

# 24. Git管理対象外

以下は`.gitignore`へ追加する。

```text
outputs/
checkpoints/
wandb/
tensorboard/
replay_buffers/
*.mcap
*.db3
*.npz
*.msgpack
*.pt
*.pth
__pycache__/
.pytest_cache/
.ruff_cache/
.venv/
```

ただし、最終提出用の小型モデルだけは次の場所へのコミットを許可する。

```text
aichallenge/workspace/src/aichallenge_submit/
└── lidar_racing_controller/models/policy_torch.pt
```

学習途中のcheckpointはコミットしない。

---

# 25. Makefileターゲット

既存ターゲットと衝突しない名前で追加する。

```makefile
.PHONY: lidar-rl-setup lidar-rl-test lidar-rl-benchmark \
        lidar-rl-train-step1 lidar-rl-train-step2 \
        lidar-rl-eval lidar-rl-export

lidar-rl-setup:
	cd aichallenge/ml_workspace/lidar_racing_rl && uv sync

lidar-rl-test:
	cd aichallenge/ml_workspace/lidar_racing_rl && uv run pytest

lidar-rl-benchmark:
	cd aichallenge/ml_workspace/lidar_racing_rl && \
	uv run python scripts/benchmark_env.py

lidar-rl-train-step1:
	cd aichallenge/ml_workspace/lidar_racing_rl && \
	uv run python scripts/train.py \
	--config-name step1_single_vehicle

lidar-rl-train-step2:
	cd aichallenge/ml_workspace/lidar_racing_rl && \
	uv run python scripts/train.py \
	--config-name step2_four_vehicle

lidar-rl-eval:
	cd aichallenge/ml_workspace/lidar_racing_rl && \
	uv run python scripts/evaluate.py

lidar-rl-export:
	cd aichallenge/ml_workspace/lidar_racing_rl && \
	uv run python scripts/export_policy.py
```

---

# 26. 実装マイルストーン

## M0：リポジトリ初期化

成果物：

* 作業ブランチ
* プロジェクトディレクトリ
* `pyproject.toml`
* `uv.lock`
* README
* submodule
* THIRD_PARTY_NOTICES
* CI smoke test

完了条件：

```text
uv sync成功
pytest実行成功
submodule初期化成功
```

## M1：単一環境・単車両

成果物：

* AI Challenge車両設定
* LiDAR-only observation
* LiDAR前処理
* frame stack
* 単車両reward
* benchmark

完了条件：

```text
単独Pure Pursuitが複数周走行
Actor観測にGTが含まれない
```

## M2：動的車両LiDAR

成果物：

* Ray–OBB
* Dynamic Scan
* occlusion
* geometry test
* fork側変更

完了条件：

```text
他車両がLiDARへ正しく映る
壁の後ろの車両が映らない
64環境でJIT実行できる
```

## M3：NPC 3台

成果物：

* JAX Pure Pursuit
* waypoint line randomization
* speed randomization
* longitudinal following control
* 4台reset sampler

完了条件：

```text
1環境4台が安定走行
NPC同士の不自然な追突が抑制される
```

## M4：SAC

成果物：

* Actor
* Twin Critic
* Replay Buffer
* entropy tuning
* collector
* checkpoint
* evaluation

完了条件：

```text
Step 1 smoke training成功
checkpoint再開成功
```

## M5：追従・追い抜き

成果物：

* relative progress
* pass判定
* overtaking scenarios
* curriculum
* opponent pool interface

完了条件：

```text
追従と追い抜き指標を自動集計可能
```

## M6：AWSIM移植

成果物：

* LiDAR rosbag解析
* domain randomization設定
* Flax→PyTorch変換
* ROS 2 inference node
* launch/config/model
* inference latency test

完了条件：

```text
make autoware-build成功
AWSIM LiDAR受信成功
control_cmd Publish成功
Flax・PyTorch出力一致
```

---

# 27. Code Agentへの実装規則

Code Agentは以下を厳守する。

1. `e2e-dev`から`feat/e2e-lidar-sac-jax`を作成する。
2. 既存`reinforcement_learning`を削除・移動・大規模変更しない。
3. 既存のRSU、TinyLiDARNet、PilotNet関連コードを変更しない。
4. ルート`requirements.txt`へJAX依存を追加しない。
5. 外部リポジトリを親リポジトリへ直接コピーしない。
6. F1TENTH Gym JAXを変更する前にforkを作成する。
7. submodule内でdetached HEADのままコミットしない。
8. 外部リポジトリのcommit SHAとライセンスを記録する。
9. Actor・CriticへGT情報を渡さない。
10. GT情報を使用する箇所にはコメントで用途を明記する。
11. 環境軸、車両軸、beam軸をPythonループで処理しない。
12. `terminated`と`truncated`を分ける。
13. EgoのtransitionだけをReplay Bufferへ保存する。
14. 設定値をコードへハードコードせずYAMLへ置く。
15. 各マイルストーンごとにテストを追加する。
16. 学習データ、Replay Buffer、通常checkpointをコミットしない。
17. 最終モデルはManifestとともに出力する。
18. READMEへ再現可能な実行コマンドを記載する。
19. 実装上の判断を`docs/decisions/`へADRとして残す。
20. 競技ルールは変更され得るため、センサ境界を設定で緩めず、LiDAR-onlyをコード上でもテストする。

---

# 28. Code Agentの初回作業範囲

最初の実装では、SAC全体を一度に完成させず、以下までを実施する。

```text
1. ブランチ作成
2. lidar_racing_rlの骨格作成
3. uv環境作成
4. fork/submodule準備
5. 単車両F1TENTH Gym JAX rollout
6. LiDAR-only observation wrapper
7. 4台環境のreset
8. JAX Pure Pursuit NPC
9. Dynamic Ray–OBB LiDAR
10. 64環境benchmark
11. Unit Test
12. READMEと設計記録
```

初回Pull Requestでは、本格的なSAC学習性能やAWSIM推論ノードを完了条件にしない。

初回Pull Requestの最終報告には以下を含める。

```text
変更ファイル一覧
親リポジトリcommit SHA
submodule commit SHA
実行コマンド
pytest結果
benchmark結果
既知の問題
次マイルストーン
```

---

# 29. Definition of Done

プロジェクト全体の完了条件は以下とする。

* LiDAR-only Actorが単独で周回できる。
* 1環境4台を複数環境並列で学習できる。
* 動的車両がLiDARに正しく映る。
* NPC 3台が異なるPure Pursuit方策で走行する。
* Ego-only SACが安定して更新される。
* 追従・追い抜き評価が自動化されている。
* AWSIMのLiDAR前処理と学習側前処理が一致している。
* AWSIMの壁抜け・欠損を学習時に再現できる。
* JAX ActorをPyTorchへ変換できる。
* FlaxとPyTorchの出力差が`1e-5`以下である。
* ROS 2ノードがLiDARを受けて制御指令をPublishできる。
* ROS 2ノードが禁止されたGT・自己位置トピックを購読しない。
* LiDAR停止時にフェイルセーフが働く。
* `create_submit_file.bash`でモデルを含む提出物を生成できる。
* 外部コードのfork、submodule、ライセンス、commit SHAが追跡可能である。
* 学習設定、seed、コードバージョンから実験を再現できる。
