# stuck_recovery_controller

MPC の指令を監視し、壁などに押し付けられて進めない場合に短時間の後退操作を行う
ROS 2 ノードです。AWSIM 実行時の MPC モードで既定で有効になり、実車モードでは既定で無効です。

## Command arbitration

復帰機能を有効にすると、MPC の出力は
`/control/command/control_cmd_mpc` にリマップされます。このノードだけが最終的な
`/control/command/control_cmd` を publish するため、MPC と復帰指令が競合しません。

通常時は MPC 指令をそのまま転送します。復帰中は次の状態を遷移します。

```text
NORMAL
  -> STOPPING
  -> SHIFT_REVERSE
  -> REVERSING
  -> STOPPING_REVERSE
  -> SHIFT_DRIVE
  -> NORMAL
```

`/control/recovery/state` (`std_msgs/msg/String`) で現在状態を確認できます。

## Stuck detection

以下をすべて満たす状態が `stuck_timeout` 継続すると復帰を開始します。

- 車速レポートが新しい
- 実車速の絶対値が `stuck_velocity_threshold` 以下
- 新しいMPC指令の目標速度・加速度が閾値以上、またはMPCがinfeasibleを通知

正の加速度条件を加えることで、MPC が障害物に対して意図的に減速・停止している場面を
復帰対象から除外しやすくしています。

## Topics

| Direction | Topic | Type |
|---|---|---|
| Subscribe | `/control/command/control_cmd_mpc` | `AckermannControlCommand` |
| Subscribe | `/control/mpc/infeasible` | `std_msgs/msg/Bool` |
| Subscribe | `/vehicle/status/velocity_status` | `VelocityReport` |
| Subscribe | `/vehicle/status/gear_status` | `GearReport` |
| Subscribe | `/localization/kinematic_state` | `nav_msgs/msg/Odometry` |
| Subscribe | `/control/recovery/trigger` | `std_msgs/msg/Empty` |
| Publish | `/control/command/control_cmd` | `AckermannControlCommand` |
| Publish | `/control/command/gear_cmd` | `GearCommand` |
| Publish | `/control/recovery/state` | `std_msgs/msg/String` |

## Manual test

手動で復帰シーケンスを開始できます。

```bash
ros2 topic pub --once /control/recovery/trigger std_msgs/msg/Empty '{}'
ros2 topic echo /control/recovery/state
ros2 topic echo /vehicle/status/gear_status
```

復帰を無効化して従来どおり MPC から直接 publish する場合は、launch に
`enable_stuck_recovery:=false` を指定します。

主要パラメータは `config/stuck_recovery.param.yaml` にあります。まず調整する値は
`stuck_timeout`、`reverse_duration`、`reverse_distance`、`cooldown_duration` です。
既定では最大 2.5 秒、または 1.5 m 後退した時点で停止して Drive に戻ります。
