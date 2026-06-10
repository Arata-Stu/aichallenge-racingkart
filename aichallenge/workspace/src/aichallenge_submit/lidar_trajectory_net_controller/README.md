# LiDAR Trajectory Net Controller

2本のvirtual LaserScanから3 channel入力を作り、PyTorchのLiDAR Trajectory Netで
将来pathを推論するROS 2 packageです。

## Input

```text
input/scan                 sensor_msgs/msg/LaserScan
input/scan_with_obstacles  sensor_msgs/msg/LaserScan
input/odometry             nav_msgs/msg/Odometry
```

## Output

```text
output/path        autoware_auto_planning_msgs/msg/PathWithLaneId
output/debug_path  nav_msgs/msg/Path
```

モデルのBezier出力はego座標です。Odometryの姿勢を使ってmap座標へ変換してから
`PathWithLaneId`としてpublishします。制御commandはpublishしません。

## Build

```bash
cd /aichallenge/workspace
colcon build --symlink-install \
  --packages-select lidar_trajectory_net_controller
source install/setup.bash
```

## Run

```bash
ros2 launch lidar_trajectory_net_controller lidar_trajectory_net.launch.xml \
  ckpt_path:=/aichallenge/ml_workspace/lidar_trajectory_net/checkpoints/best_model.pth \
  device:=auto
```

新形式checkpointでは学習時のmodel/data設定を自動的に使用します。
`model.use_checkpoint_config:=false`にするとROS parameter側の設定を使用します。

必要になった段階で、既存の`path_to_trajectory`を別途接続できます。

```bash
ros2 run path_to_trajectory path_to_trajectory_node \
  --ros-args \
  -r input:=/planning/lidar_trajectory_net/path \
  -r output:=/planning/lidar_trajectory_net/trajectory
```
