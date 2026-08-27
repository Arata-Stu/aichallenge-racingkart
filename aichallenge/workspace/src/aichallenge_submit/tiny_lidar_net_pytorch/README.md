# tiny_lidar_net_pytorch (ROS 2 inference)

TinyLiDARNetをPyTorchで実行するROS 2推論専用パッケージです。学習コードは
`aichallenge/ml_workspace/tiny_lidar_net_pytorch`に分離しています。学習側が保存した
`best_model.pth`を直接読み込むため、`.npy/.npz`への重み変換は不要です。

旧`tiny_lidar_net_controller`（NumPy推論）は変更せず残しています。

## Build

```bash
cd /aichallenge/workspace
colcon build --symlink-install --packages-select tiny_lidar_net_pytorch aichallenge_submit_launch
source install/setup.bash
```

## ROS 2推論

```bash
ros2 launch tiny_lidar_net_pytorch tiny_lidar_net_pytorch.launch.xml \
  checkpoint_path:=/path/to/checkpoints/best_model.pth device:=auto
```

または環境変数を設定し、標準launchの制御方式として起動できます。

```bash
export TINY_LIDAR_NET_PYTORCH_CHECKPOINT=/path/to/checkpoints/best_model.pth
ros2 launch aichallenge_submit_launch reference.launch.xml \
  simulation:=true use_sim_time:=true control_method:=tiny_lidar_net_pytorch
```

`device:=auto`はCUDAが利用可能ならGPU、利用不可ならCPUを選択します。明示的にGPUを必須にする場合は
`device:=cuda`を指定してください。

標準設定ではTinyLiDARNetの推論値のうちステアだけを使用します。アクセルはルールベースで、
発進から15 km/h付近までは`1.0`、その後は`0.7`です。14〜16 km/hのヒステリシスを持たせ、
速度トピックが一時的に来ない場合も`0.7`を出すため、未学習のアクセル推論値で停止することはありません。

チェックポイントを番号選択して単独のPlayerを起動する場合：

```bash
/aichallenge/utils/run_tiny_lidar_player.bash 2
```

Player 2〜4を同時に走行させる場合は`/aichallenge/utils/run_terminator.bash`を起動し、各Player
ペインの`aic_player_menu`を実行して`tiny`とcheckpointを選択します。選択はPlayerごと・実行ごとに
行うため、Terminatorを開いたまま異なる重みを比較できます。全Playerペインを先に起動し、
AWSIMペインは最後に起動してください。
AWSIM開始後にPlayerを起動すると、一度だけ通知される`Ready/Grounded`状態を取り逃して初期化待ちに
なることがあります。
