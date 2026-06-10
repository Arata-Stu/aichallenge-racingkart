# ml_workspace

機械学習（ML）関連の作業用ディレクトリです。データ収集（rosbag記録）や、学習、学習済み重み管理などの補助スクリプトを置きます。

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
   ├─ README.md
   ├─ requirements.txt
   ├─ train.py
   ├─ config/
   │  └─ train.yaml
   ├─ datasets/             # extract_data_from_bag.py の出力先（例）
   │  ├─ train/...
   │  └─ val/...
   ├─ lib/
   │  ├─ __init__.py
   │  ├─ data.py
   │  ├─ loss.py
   │  └─ model.py
   ├─ outputs/              # 学習ログ出力先（Hydraの既定）
   ├─ extract_data_from_bag.py
   └─ osm2csv.py
└─ lidar_trajectory_net/
   ├─ README.md
   ├─ requirements.txt
   ├─ train.py
   ├─ extract_data_from_bag.py
   ├─ config/
   │  └─ train.yaml
   └─ lib/
      ├─ __init__.py
      ├─ data.py
      ├─ loss.py
      └─ model.py
```

## 各項目の説明

- `.gitignore`: 学習データや生成物をリポジトリに含めないための設定です。
- `record_data.bash`: 学習用データ作成のために rosbag（mcap）を `rawdata/` 配下へ記録する補助スクリプトです。
- `rawdata/`: 記録した rosbag（mcap）の保存先です（タイムスタンプ名のディレクトリが作られます）。
- `train/`, `val/`: `rawdata/` から分けた rosbag（mcap）を置くためのディレクトリです（運用に応じて使います）。
- `tiny_lidar_net/`: TinyLiDARNet 用のデータ変換・学習コード一式です。PyTorchの`.pth`をROS 2側で直接利用します。使い方は `aichallenge/ml_workspace/tiny_lidar_net/README.md` を参照してください。
- `lidar_trajectory_net/`: virtual LiDAR 2本と差分を入力し、時系列 Transformer + Bezier head で ego frame の将来 path を予測する学習コードです。使い方は `aichallenge/ml_workspace/lidar_trajectory_net/README.md` を参照してください。
