# rsu_fusion_net

This workspace trains an ego-priority RSU LiDAR fusion policy from rosbag-derived
NumPy arrays.

The model uses:

1. per-frame 1D Conv scan encoding,
2. GRU temporal encoding over multiple scan frames,
3. distance-gated RSU fusion with optional top-k selection,
4. a small MLP control head.

## Dataset format

Each sequence directory must contain:

- `ego_scans.npy`: `[N, R]`
- `rsu_scans.npy`: `[N, S, R]`
- `rsu_meta.npy`: `[N, S, M]`
- `targets.npy`: `[N, 2]`

Optional:

- `vehicle_state.npy`: `[N, V]`
- `rsu_mask.npy`: `[N, S]`

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

## Compare history lengths

```bash
cd /aichallenge/ml_workspace/rsu_fusion_net
HISTORY_LENGTHS="1 3 5 8 10" bash scripts/compare_history_lengths.sh
```

## Rosbag preprocessing

Use `preprocess_bag_to_npy.py` inside a ROS 2 environment. It reads ego scan,
multiple RSU scan topics, and control commands, then writes the dataset files.
Topic names are CLI arguments so this stays independent from launch files.

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

`--rsu-meta` supplies static RSU metadata. The preprocessor appends `age_s`,
so the saved `rsu_meta.npy` layout is:

```text
[distance_m, relative_x_m, relative_y_m, relative_yaw_rad, age_s]
```
