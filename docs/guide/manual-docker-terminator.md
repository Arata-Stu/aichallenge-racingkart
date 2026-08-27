# Docker + Terminator manual development

This workflow prepares AWSIM and one to four player Autoware stacks in a single
Docker container. AWSIM uses ROS domain 0. Player N uses ROS domain N.

## Start

Enter the development container from the host:

```bash
make autoware-bash
```

Then start the predefined Terminator layout inside the container:

```bash
/aichallenge/utils/run_terminator.bash
```

The launcher interactively asks for the number of players. Enter accepts the
default of three:

```text
Number of players [1-4] (default: 3):
Gamepad player [0-3] (0: none, default: 0):
RViz player [1-3] (default: 1):
```

These three prompts only decide the window layout and assignments. Simulator
mode, control method, checkpoint, run mode, and control parameters are deliberately
not fixed while Terminator is opening.

The selected layout contains AWSIM, Bag Manager, a free ROS 2 shell, and session
control at the top, with one Autoware pane per player at the bottom. It does not
start AWSIM, Bag Manager, or Autoware until Enter is pressed. For three players:

| Pane | Process | ROS domain |
|---|---|---:|
| Top left | Prepared AWSIM command (`dev`, three vehicles) | 0 |
| Top center-left | Prepared `aic_bag_manager` command | Gamepad player |
| Top center-right | Free ROS 2 shell | 1 by default |
| Top right | Prepared `aic_stop_all` session control command | — |
| Bottom left | Prepared Player 1 Autoware and RViz command | 1 |
| Bottom center | Prepared Player 2 Autoware and RViz command | 2 |
| Bottom right | Prepared Player 3 Autoware and RViz command | 3 |

The AWSIM pane initially contains `aic_simulator_menu`, and every Player pane
contains `aic_player_menu`. Press Enter in a Player pane to select its settings:

```text
Player 2 control:
  1) Joycon
  2) TinyLiDARNet PyTorch
  3) RSU Fusion PyTorch
  4) MPC
Select [1-4] (default: 2): 3
Run mode:
  1) awsim (RViz on)
  2) awsim-no-viz (RViz off)
  3) vehicle
  4) rosbag
Select [1-4] (default: 2):
RSU Fusion checkpoints: /aichallenge/ml_workspace/rsu_fusion_net/checkpoints
  1) versions/only-run/h5_.../best_model.pth
Select checkpoint [1-1] (default: 1):
```

The checkpoint list is discovered at that moment, so a model trained after the
Terminator window was opened can be selected without reopening the window. Each
Player selects independently. After stopping a Player, pressing Enter on
`aic_player_menu` asks again, allowing a different model or controller on the next run.

Press Enter on `aic_simulator_menu` to select the simulator mode and wall recovery
immediately before AWSIM starts. MPC obstacle avoidance is similarly selected only
when `mpc` is selected in a Player pane. These menus also use numbered choices;
pressing Enter accepts the displayed default.

The Bag
Manager pane contains `aic_bag_manager`; it automatically uses the selected
Gamepad/Joycon player's domain. Press Enter to start it, then use R1/L1 to
start/stop each recording. If no Gamepad player was selected, the pane shows a
clear error instead of recording the wrong domain. The control pane contains
`aic_stop_all`, which safely stops the Bag Manager, AWSIM, and every player
process tree together. The free shell accepts arbitrary commands such as
`ros2 topic list`. It starts on Domain 1; run `aic_domain 0` or
`aic_domain <player-number>` to switch domains in that shell. Its command
history is retained in `/output/terminator-history/free_shell_history`.
Edit a prefilled command if needed and press Enter to start it. Nothing starts merely
by opening Terminator. The commands are also saved to persistent per-domain
history files under `/output/terminator-history`, so they can be recalled with
the Up/Down keys after restarting Terminator. Once started, the upper panes
stream the logs written by the standard startup scripts. The session output
directory is printed before Terminator starts.

`gamepad_player=0` disables Gamepad control. A Gamepad-assigned Player defaults to
`joycon`, Player 1 otherwise defaults to `mpc`, and Player 2以降は`tiny`が既定です。
These are only Enter-key defaults; `aic_player_menu` can select any controller for
each run. `tiny`は`control_method=tiny_lidar_net_pytorch`、`rsu`は
`control_method=rsu_fusion_net_pytorch`、`mpc`は従来のMPCになります。
TinyLiDARNetのチェックポイントはPlayerを実行するたびに番号選択します。
各Playerは別のROS domainで自車Virtual ScanとPyTorch推論ノードを起動するため、Player 2〜4を
学習済みポリシーで同時走行させられます。Playerペインには`aic_player_menu`が入力済みで、
Enter後に制御方式とcheckpointを選びます。Gamepadに割り当てたPlayerではJoyconが既定です。

RSU Fusionでも各Playerの実行時にチェックポイントを選択します。`aic_rsu_fusion`は
自車Virtual Scan（障害物あり）と6つのRSU Scanを各ROS domainで生成します。
旧ステア専用checkpointはアクセル2.0固定、新しいアクセル学習済みcheckpointはアクセルとステアの
両方を自動推論します。

Joycon Playerの`/sensing/lidar/scan`には、`/v2x/vehicle_positions`から得た他車両を矩形障害物として
反映します。同時に`/sensing/lidar/scan_without_obstacles`へ静的なコース境界だけのScanを配信します。
TinyLiDARNet NPCは後者を明示的に使用するため、近くの車両によって入力分布が変化せず、従来どおり
安定したレーン追従を行います。Bag Managerは比較・再学習用に両方のTopicを記録します。

複数台ではPlayer 1〜Nの全ペインを先にEnterで起動し、AWSIMペインを最後に起動してください。
AWSIMを先に開始すると、遅れて起動したPlayerが一度だけ送られる`Ready/Grounded`状態を受信できず、
初期姿勢設定前で待ち続けることがあります。AWSIMペインにもこの起動順が表示されます。
Only one player can be assigned to the physical Gamepad by this launcher.
Changing control methods requires stopping that Player and running
`aic_player_menu` again; reopening Terminator is unnecessary.

Only one player launches RViz. When a Gamepad player is selected, RViz follows
that player by default; otherwise it uses Player 1. The interactive RViz prompt
can select a different player. In simulation, the Gamepad-controlled `joycon`
player launches the ego Virtual Scan and shared RSU LaserScan generators.
Each `tiny_lidar_net_pytorch` NPC launches only its own ego Virtual Scan on its
separate ROS domain; it does not launch duplicate RSU generators.

MPC obstacle avoidance is enabled by default in this manual workflow. When
enabled, `multi_purpose_mpc_ros` subscribes to `/v2x/vehicle_positions` and
uses predicted V2X vehicle positions as dynamic obstacles.

For simulator modes that provide `--wall-recovery`, the launcher asks whether
AWSIM wall recovery should be `off` or `on`. The default remains `off`, matching
the normal development and evaluation scripts. This is an AWSIM-side recovery
feature, not an autonomous reverse maneuver generated by MPC.

## Stop

Press `Ctrl+C` in an upper pane to stop AWSIM or Autoware. The pane remains
open and returns to the prefilled command prompt, so Enter can start the
process again. Pressing `Ctrl+C` before starting a command also keeps the pane
open. Run `aic_stop_all` in the control pane to stop the Bag Manager, AWSIM,
and all players at once. Close the Terminator window when the entire manual
session is finished.

The Terminator configuration is stored at
`/aichallenge/utils/terminator.config` and can be edited without changing the
Docker image or the user's temporary home directory.
