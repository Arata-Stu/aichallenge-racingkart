# rsu_fusion_net_pytorch

`ml_workspace/rsu_fusion_net`で学習した`best_model.pth`を変換せずに読み込むROS 2推論パッケージです。
自車Virtual Scan、6つのRSU Scan、自己位置、自車速度を学習時と同じ履歴形式にし、
複数候補の速度付き将来軌道、候補確率、アクセル、ステアを推論します。

```bash
colcon build --symlink-install --packages-select rsu_fusion_net_pytorch aichallenge_submit_launch
source install/setup.bash
export RSU_FUSION_NET_PYTORCH_CHECKPOINT=/path/to/best_model.pth
ros2 launch rsu_fusion_net_pytorch rsu_fusion_net_pytorch.launch.xml
```

通常はパスを入力せず、次のスクリプトを使います。

```bash
/aichallenge/utils/run_rsu_fusion_player.bash 2 awsim-no-viz
```

通常の`control_mode:=ai`ではアクセルとステアの両方を学習済みモデルから出力します。
`auto`も固定アクセルへフォールバックしません。`fixed_full`モードは使用できません。

軌道checkpointでは次も配信します。

- `~/selected_trajectory` (`nav_msgs/Path`)
- `~/candidate_trajectories` (`visualization_msgs/MarkerArray`)
- `~/mode_probabilities` (`std_msgs/Float32MultiArray`)

`autoware-rsu.rviz`には候補軌道Markerの表示設定を追加済みです。
