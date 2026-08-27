# Policy artifacts

最終exportで生成した次の2ファイルをこのディレクトリへ配置します。現在はweightを同梱していません。ファイルが無い状態でもノードは起動し、設定した安全停止指令を20 Hzでpublishします。

```text
policy_torch.pt
policy_manifest.json
```

`policy_torch.pt`はpickle任意オブジェクトではなく、`LidarActor.state_dict()`そのもの、または`{"state_dict": state_dict}`だけを保存します。ノードは`torch.load(..., weights_only=True)`で読み込み、キーとshapeを厳密検証します。

manifestの最小スキーマ:

```json
{
  "architecture_version": "lidar_actor_conv1d_v1",
  "architecture": {
    "conv_channels": [32, 64, 64],
    "kernel_sizes": [8, 4, 3],
    "strides": [4, 2, 1],
    "hidden_dim": 256,
    "action_dim": 2,
    "log_std_min": -5.0,
    "log_std_max": 2.0
  },
  "beam_count": 360,
  "frame_stack": 4,
  "scan_channels": 2,
  "field_of_view": 4.71238898038469,
  "range_normalization": {
    "type": "divide_by_range_max",
    "range_max": 30.0,
    "output_min": 0.0,
    "output_max": 1.0
  },
  "validity": {"valid": 1.0, "invalid": 0.0},
  "action_scaling": {
    "steering_max_abs": 0.64,
    "acceleration_min": -3.2,
    "acceleration_max": 3.2
  },
  "training_config_hash": "<sha256>",
  "root_repository_commit": "<commit-sha>",
  "f1tenth_gym_jax_commit": "<commit-sha>",
  "model_checksum": {"algorithm": "sha256", "value": "<sha256>"},
  "export_timestamp": "<ISO-8601>"
}
```

beam数、frame stack、FOV、入力range max、action scaling、SHA-256のいずれかがランタイム設定と一致しなければ、モデルは有効化されません。
