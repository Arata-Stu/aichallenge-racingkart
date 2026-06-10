# LiDAR Trajectory Net Workspace

Virtual LiDAR の時系列から ego frame の将来 path を予測するモデルです。

入力は以下の3 channelです。

```text
0: /sensing/virtual_lidar/scan
1: /sensing/virtual_lidar/scan_with_obstacles
2: diff = max(scan - scan_with_obstacles, 0)
```

モデルは各時刻の scan を 1D Conv encoder で特徴量化し、その時系列 token を Transformer で処理します。出力 head は Bezier control points を予測し、そこから滑らかな path 点列を sample します。

## データ抽出

`awsim-joycon` などで記録した rosbag から、学習用 `.npy` を作ります。

```bash
python3 extract_data_from_bag.py \
  --bags-dir /path/to/rosbags \
  --outdir ./dataset/train
```

既定 topic は以下です。

```text
/sensing/virtual_lidar/scan
/sensing/virtual_lidar/scan_with_obstacles
/localization/kinematic_state
```

出力 sequence には以下が保存されます。

```text
scan_inputs.npy  # [N, 3, num_rays]
poses.npy        # [N, 3] = [x, y, yaw]
timestamps.npy
metadata.json
```

## 学習

```bash
python3 train.py \
  data.train_dir=/path/to/train \
  data.val_dir=/path/to/val \
  data.history_length=8 \
  data.future_num_points=20 \
  data.future_stride=2 \
  train.device=auto
```

`future_num_points` と `future_stride` を変えることで、教師 path の点数と時間方向の間隔を調整できます。たとえば generator が 50 Hz なら、`future_stride=5` は 0.1 秒間隔です。

## Shape

Dataset が返す tensor は以下です。

```text
input : [T, 3, num_rays]
target: [future_num_points, 2]  # ego frame path [x, y]
```

DataLoader 後は batch 次元がついて以下になります。

```text
input : [B, T, 3, num_rays]
target: [B, future_num_points, 2]
```

## Model

既定構成は軽めです。

```text
1D Conv encoder: 3ch -> 32 -> 64 -> 128
Temporal Transformer: 2 layers, 4 heads
Bezier head: P0=(0,0) fixed, P1..P3 predicted
```

まずはこの設定で動かし、必要なら `config/train.yaml` の `history_length`, `future_num_points`, `embed_dim`, `transformer_layers` を調整してください。
