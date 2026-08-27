# ml_workspace

機械学習（ML）関連の作業用ディレクトリです。データ収集（rosbag記録）や、学習・重み変換などの補助スクリプトを置きます。

## ディレクトリ構成

```text
ml_workspace/
├─ README.md
├─ .gitignore
├─ record_data.bash
├─ rawdata/                 # rosbag（生データ）保存先
│  └─ YYYYMMDD-HHMMSS/...
├─ train/                   # 学習用に分けたrosbag置き場（任意）
│  └─ YYYYMMDD-HHMMSS/...
├─ val/                     # 検証用に分けたrosbag置き場（任意）
│  └─ YYYYMMDD-HHMMSS/...
├─ tiny_lidar_net/
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ train.py
│  ├─ config/
│  │  └─ train.yaml
│  ├─ datasets/             # extract_data_from_bag.py の出力先（実行時に生成；コミットされない）
│  │  ├─ train/...
│  │  └─ val/...
│  ├─ lib/
│  │  ├─ __init__.py
│  │  ├─ data.py
│  │  ├─ loss.py
│  │  └─ model.py
│  ├─ outputs/              # 学習ログ出力先（Hydraの既定；実行時に生成；コミットされない）
│  ├─ extract_data_from_bag.py
│  ├─ osm2csv.py
│  └─ convert_weight.py
├─ tiny_lidar_net_pytorch/       # PyTorch学習専用（.pthをROS 2側から直接利用）
│  ├─ README.md
│  ├─ train.py
│  └─ lib/
├─ pilot_net/               # PilotNet 用のデータ変換・学習コード一式
├─ joy_profile_editor/      # /dev/input/js0 実測ベースの teleop_manager YAML 生成GUI
├─ rsu_fusion_net/          # 車両LiDAR + RSU LiDAR の時系列ゲート融合モデル
├─ reinforcement_learning/  # 公式の画像ベース強化学習コード
└─ virtual_scan_rl/         # Virtual Scan + TinyLiDAR CNN + SAC（Joy介入対応）
```

## 各項目の説明

- `.gitignore`: 学習データや生成物をリポジトリに含めないための設定です。
- `record_data.bash`: 学習用データ作成のために rosbag（mcap）を `rawdata/` 配下へ記録する補助スクリプトです。
- `rawdata/`: 記録した rosbag（mcap）の保存先です（タイムスタンプ名のディレクトリが作られます）。
- `train/`, `val/`: `rawdata/` から分けた rosbag（mcap）を置くためのディレクトリです（運用に応じて使います）。
- `tiny_lidar_net/`: TinyLiDARNet 用のデータ変換・学習・重み変換コード一式です。使い方は `aichallenge/ml_workspace/tiny_lidar_net/README.md` を参照してください。
- `tiny_lidar_net_pytorch/`: 新しいTinyLiDARNetのPyTorch学習専用コードです。`run_pipeline.sh`で録画選択・train/val前処理・GPU学習を対話的に実行でき、`.pth`を変換せずROS 2推論パッケージから読み込みます。
- `pilot_net/`: PilotNet 用のデータ変換・学習コード一式です。
- `joy_profile_editor/`: DualShock4 などの `/dev/input/js0` 実測値から `teleop_manager` 用 YAML を生成する小さなWeb UIです。
- `rsu_fusion_net/`: 複数時刻の車両 LiDAR と RSU LiDAR を 1D Conv + GRU + distance gate で融合し、複数候補の速度付き将来軌道、候補確率、アクセル、ステアを同時学習します。Web UIで前処理・可視化・学習・オフライン評価を行い、生成した`.pth`をROS 2推論から直接使用します。
- `reinforcement_learning/`: 強化学習用の学習・評価コード一式です。
- `virtual_scan_rl/`: 公式コードを変更せず分離したVirtual Scan SAC環境です。まず単車完走を学習し、後からNPC追い越しへ拡張します。Joyを押している間の人間介入と、実行行動のReplay Buffer保存に対応します。
