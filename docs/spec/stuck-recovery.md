# MPC スタック自動復帰

## 目的

MPC が壁や他車との接触後に前進指令を出し続けても車速が上がらない場合、車両を停止して
REVERSE に切り替え、短距離後退した後に DRIVE と通常の MPC 制御へ復帰する。

## 制御コマンドの所有権

復帰中だけ別ノードから同じトピックへ publish すると、40 Hz の MPC 指令と競合して
AWSIM がどちらを採用するか不定になる。このため、復帰機能が有効な場合は次の構成にする。

```text
mpc_controller
  /control/command/control_cmd_mpc
                 |
                 v
stuck_recovery_controller
  |-- /control/command/control_cmd    (唯一の最終 Publisher)
  `-- /control/command/gear_cmd
```

`NORMAL` では MPC 指令を透過する。その他の状態では停止または後退指令を出し、MPC の
計算自体は継続させる。DRIVE 復帰後は最新の MPC 指令へ戻るため、MPC の再起動は不要。

## 状態遷移

| State | Control command | Gear command | Exit condition |
|---|---|---|---|
| `NORMAL` | MPC を透過 | DRIVE | スタック継続、または手動トリガ |
| `STOPPING` | 停止 | DRIVE | 車速ゼロを一定時間確認 |
| `SHIFT_REVERSE` | 停止 | REVERSE | ギア確認、またはフィードバックなし時のタイムアウト |
| `REVERSING` | 負の速度、正の加速度 | REVERSE | 後退距離または後退時間に到達 |
| `STOPPING_REVERSE` | 停止 | REVERSE | 車速ゼロを一定時間確認 |
| `SHIFT_DRIVE` | 停止 | DRIVE | ギア確認、またはフィードバックなし時のタイムアウト |

明示的に異なるギアのフィードバックを受けている場合、タイムアウトだけで次の走行状態へは
進まない。ギア状態が取得できない車両インターフェースの場合のみ時間で代替する。

## スタック判定

既定値では、次の状態が 2.5 秒続くとスタックと判定する。

- `abs(longitudinal_velocity) <= 0.20 m/s`
- MPC の `longitudinal.speed >= 1.0 m/s` かつ
  `longitudinal.acceleration >= 0.30 m/s^2`、または
  `/control/mpc/infeasible` が直近に `true`

正の加速度条件により、MPC が他車や障害物を認識して意図的に制動している場合を
復帰対象から除外しやすくする。一方、経路制約またはソルバがinfeasibleになってMPCが
停止指令へ落ちた場合は、明示的なinfeasible通知を使って検出を継続する。
起動直後は5秒、復帰直後は4秒の抑制時間を設ける。

## 複数台走行

ノードは各車両の ROS Domain（1〜4）内で起動する。トピック名は共通だが DDS Domain が
分かれているため、スタック判定、ギア指令、復帰状態は車両ごとに独立する。

後方障害物を直接判定するセンサ条件は現時点では含めていない。実走では1.5秒の旧設定が
毎回距離到達前に終了していたため、後退は既定で最大2.5秒または1.5 mに制限する。
密集時はこの値を小さくし、実走ログに合わせて調整する。

## 有効化

AWSIM (`simulation:=true`) かつ MPC モードでは既定で有効。実車モードでは安全のため
既定で無効。明示的に切り替える場合は次の launch 引数を使う。

```text
enable_stuck_recovery:=true|false
```

状態確認と手動試験:

```bash
ros2 topic echo /control/recovery/state
ros2 topic pub --once /control/recovery/trigger std_msgs/msg/Empty '{}'
ros2 topic echo /vehicle/status/gear_status
```
