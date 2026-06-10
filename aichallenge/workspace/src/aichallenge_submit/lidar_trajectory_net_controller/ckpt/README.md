# Checkpoint

学習済みcheckpointを直接指定して起動します。

```bash
ros2 launch lidar_trajectory_net_controller lidar_trajectory_net.launch.xml \
  ckpt_path:=/aichallenge/ml_workspace/lidar_trajectory_net/checkpoints/best_model.pth
```

新形式checkpointはmodel/data設定を内包します。重みのみの旧形式checkpointを使う場合は、
`config/lidar_trajectory_net.param.yaml`のmodel parameterを学習時と一致させてください。
