# rsu_fusion_net

This workspace trains an ego-priority RSU LiDAR fusion policy from rosbag-derived
NumPy arrays.

Two training inputs can be selected from the dashboard:

- `Semantic BEV`: an efficient 2D depthwise CNN and GRU consume the map/wall,
  free-space, dynamic-obstacle and occlusion channels generated onboard.
- `Ego + RSU Scan`: the existing separate 1D scan encoders and distance-gated fusion.

Both models use:

1. per-frame 1D Conv scan encoding,
2. GRU temporal encoding over multiple scan frames,
3. distance-gated RSU fusion with optional top-k selection,
4. ordered 0.75-second anchors decoded into a piecewise cubic Bezier path,
5. a fixed 3-second horizon sampled as 12 waypoints at 0.25-second intervals,
6. learned mode probabilities and a direct acceleration/steering head.

The Bezier decoder generates each anchor from a bounded forward step and heading
change, then joins the anchors with shared tangents. It therefore cannot emit the
unrelated long point-to-point jumps possible with the legacy direct waypoint head.
Training explicitly optimizes both average displacement error (ADE) and final
displacement error (FDE). Legacy `trajectory_multimodal` checkpoints remain usable
for comparison, but they are not converted into Bezier checkpoints.

## Dataset format

Each sequence directory must contain:

- `ego_scans.npy`: `[N, R]`
- `rsu_scans.npy`: `[N, S, R]`
- `rsu_meta.npy`: `[N, S, M]`
- `targets.npy`: `[N, 2]`

BEV preprocessing additionally writes:

- `bev_frames.npy`: packed `uint8 [N, H, W]`; bits 0-7 correspond to the eight
  channels of `/perception/virtual_scan_bev/image`

One packed byte represents all eight binary channels for a cell. This is about
one eighth of the uncompressed `8UC8` tensor size. The PyTorch Dataset expands
the selected channels only for the history frames in the current batch.
The current training size is `140 x 160` (rear 8 m, front 20 m, 16 m per side).
Older front-32 m recordings are cropped from their far-forward edge to this
same coordinate-aligned size, so old and new sequences can share a Version.

Optional:

- `vehicle_state.npy`: `[N, V]`
- `rsu_mask.npy`: `[N, S]`

Trajectory training additionally requires files written by the current preprocessor:

- `ego_poses.npy`: `[N, 3]` map-frame `(x, y, yaw)`
- `timestamps_ns.npy`: `[N]`
- `vehicle_state.npy`: `[N, 1]` ego speed in m/s

Datasets made by an older preprocessor must be processed again before trajectory
training. The future pose is transformed into the current ego frame at load time.

`rsu_meta[..., 0]` is treated as distance in meters for the distance gate.
Suggested meta layout is:

```text
[distance_m, relative_x_m, relative_y_m, relative_yaw_rad, age_s]
```

## Train

```bash
cd /aichallenge/ml_workspace/rsu_fusion_net
python3 train.py data.history_len=5
```

## Web UI

録画選択、Dataset Version、前処理、Ego/RSU同期データの可視化、学習、PID管理された
ジョブ停止をブラウザから操作できます。リポジトリのホスト側で次を実行します。

「収集メモ」では、R1/L1で停止した最新録画が自動選択され、追従・追い越しなどのカテゴリ、
成功・失敗、学習への採用可否、複数Dataset Version、自由記述メモを保存できます。
メモはTinyLiDAR Studioと共通の`collection_annotation.json`として各Bag内に保存されます。

```bash
make rsu-fusion-dashboard
```

ブラウザで `http://localhost:8766` を開きます。TinyLiDAR Studio (`8765`) と同時起動できます。
Joycon録画では自車Scanを障害物あり／なしから選択でき、Bag Managerが記録した6つのRSU Topicは
自動設定されます。自己位置と固定RSU姿勢から各フレームの距離・自車座標系での相対位置・相対角度も
自動計算されます。学習はJoyconの操作を教師として、複数の速度付き将来軌道、候補確率、
アクセル、ステアを同時に対象にします。

### Semantic BEV workflow

1. AutowareのRSU構成を起動します。`virtual_scan_bev_generator`が
   `/perception/virtual_scan_bev/image`を生成します。
2. Bag Managerで通常どおり録画します。packed BEV、制御指令、自己位置、速度、Ego/RSU Scanが
   同じbagへ保存されます。
3. Dashboardの「前処理」で`学習入力 = Semantic BEV`を選び、Train/Val録画をまとめて指定します。
   BEVがbagに存在しない場合はジョブを明示的に失敗させます。
4. 「学習」で`Semantic BEV (2D CNN)`を選びます。既定のBatch sizeは8です。
   GPUメモリに余裕があれば増やせます。
5. 生成されたcheckpointは同じ「オフライン評価」からADE/FDE、速度、アクセル、ステアを評価できます。

学習に使う既定チャンネルは0-5（走行可能、観測free、壁、動的障害物、障害物遮蔽、壁外）です。
可視化用の`debug_image`や、意味チャンネルを統合した`occupancy_grid`はモデルへ入力しません。

新規学習の既定値は「3秒先、評価Waypoint 0.25秒間隔、Bezierアンカー0.75秒間隔」です。
Dashboard上でこれらをまとめて変更でき、設定した予測時間とアンカー間隔もその場で表示されます。
損失ログには正規化距離のADE/FDE、オフライン評価にはメートル単位のADE/FDEが出力されます。

「オフライン評価」タブではVersion、checkpoint、Train/Valを選んで評価できます。結果には
ADE/FDE、速度MAE、アクセルMAE、ステアMAEが保存されます。「データを見る」の予測Overlayで、
全候補軌道、選択候補、教師軌道をフレームごとに重ねて確認できます。評価タブのCourse Mapでは、
コース境界、実走軌跡、ADEヒートマップ、Map座標へ戻した予測・教師軌道を確認できます。
選択軌道と教師軌道には各0.25秒のWaypointを点で表示するため、点の順序や後半の乱れも
フレーム単位で確認できます。

学習後の`best_model.pth`はROS 2側の`rsu_fusion_net_pytorch`が直接読み込みます。
複数PlayerのTerminatorでは各Playerペインの`aic_player_menu`を実行し、その場で`rsu`と
checkpointを選択します。Terminatorを開き直さず、Playerを再実行するたびに別の重みへ
変更できます。単体起動は次のコマンドです。

```bash
/aichallenge/utils/run_rsu_fusion_player.bash 2 awsim-no-viz
```

## Compare history lengths

```bash
cd /aichallenge/ml_workspace/rsu_fusion_net
HISTORY_LENGTHS="1 3 5 8 10" bash scripts/compare_history_lengths.sh
```

## Rosbag preprocessing

Use `preprocess_bag_to_npy.py` inside a ROS 2 environment. It reads ego scan,
multiple RSU scan topics, and control commands, then writes the dataset files.
Topic names are CLI arguments so this stays independent from launch files.

For normal use, start the interactive wrapper. It discovers Bag Manager
recordings under `/aichallenge/record`, lets you select a sequence and
train/validation destination, supplies all six RSU topics, and offers to open
the visualization when preprocessing finishes:

```bash
cd /aichallenge/ml_workspace/rsu_fusion_net
./preprocess_interactive.sh
```

To search a different recording directory or only list detected sequences:

```bash
./preprocess_interactive.sh --record-root /path/to/record
./preprocess_interactive.sh --record-root /path/to/record --list
```

The lower-level equivalent is shown below for automation and debugging.

Example:

```bash
python3 preprocess_bag_to_npy.py \
  --bag /aichallenge/ml_workspace/rawdata/session_001 \
  --output ./datasets/train/session_001 \
  --ego-scan-topic /sensing/lidar/scan \
  --rsu-scan-topics /rsu/curve_01/scan,/rsu/curve_02/scan \
  --control-topic /control/command/control_cmd \
  --rsu-meta "8.0,8.0,0.0,0.0;14.0,12.0,6.0,0.3"
```

BEVを必須にする場合は次を追加します。

```bash
  --bev-topic /perception/virtual_scan_bev/image --require-bev
```

`--rsu-meta` supplies static RSU metadata. The preprocessor appends `age_s`,
so the saved `rsu_meta.npy` layout is:

```text
[distance_m, relative_x_m, relative_y_m, relative_yaw_rad, age_s]
```

## Validate and visualize preprocessed data

Open the synchronized ego/RSU scans and control targets in an interactive
viewer:

```bash
cd /aichallenge/ml_workspace/rsu_fusion_net
python3 visualize_dataset.py --dataset ./datasets/train/session_001
```

Use the slider or Left/Right and PageUp/PageDown keys to move through samples.
The viewer prints array shapes, scan value statistics, per-RSU availability,
synchronization age, and control ranges before opening the window. RSU scans
masked as unavailable are dimmed.

For headless validation or CI:

```bash
python3 visualize_dataset.py \
  --dataset ./datasets/train/session_001 \
  --index 100 \
  --save ./dataset_check.png \
  --no-show

python3 visualize_dataset.py \
  --dataset ./datasets/train/session_001 \
  --report-only \
  --fail-on-warning
```

The preprocessed arrays do not retain the original LaserScan angle bounds, so
the horizontal scan axis is normalized ray position rather than a physical
angle. Distances and synchronization validity are unchanged.

## ROS 2 outputs

Trajectory checkpoints are consumed without conversion. In addition to the learned
control command, the node publishes:

- `~/selected_trajectory`: selected `nav_msgs/Path`
- `~/candidate_trajectories`: all candidates as `visualization_msgs/MarkerArray`
- `~/mode_probabilities`: learned candidate probabilities

The acceleration command is always produced by the neural network in the default
`control_mode:=ai`; there is no automatic fixed-acceleration fallback.
