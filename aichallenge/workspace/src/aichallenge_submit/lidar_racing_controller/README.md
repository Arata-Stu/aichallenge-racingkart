# lidar_racing_controller

2D LiDARだけを入力とするPyTorch ActorをAWSIM上で実行するROS 2パッケージです。

## I/O

- Subscribe: `/sensing/lidar/scan` (`sensor_msgs/msg/LaserScan`)
- Publish: `/control/command/control_cmd` (`autoware_auto_control_msgs/msg/AckermannControlCommand`)

Actor用に他のセンサ、自己位置、地図、V2X、車両状態トピックは購読しません。

## 処理

1. LaserScanのbeam数、range metadata、期待FOV（既定270°）を検証
2. 1080点を有効値minimum poolingで360点へ変換
3. rangeを`range_max`で0〜1へ正規化し、validityを別channel化
4. 4フレームを時系列順に保持し、`[1, 8, 360]`をActorへ入力
5. Actor meanを`tanh`した決定論的行動を操舵角・加速度へscale
6. 20 Hz timerでrate limit後の`AckermannControlCommand`をpublish

## フェイルセーフ

モデル／manifest未配置、checksum・shape・設定不一致、LaserScan timeout、metadata異常、有効beam不足、非有限Actor出力、推論例外のいずれかで安全停止へ移行します。安全判定の時刻とノード状態はActor入力には含めません。

安全停止はノードを終了させず、center steeringと設定可能な非正加速度をsteady-time timerからpublishし続けます。有効なscanと推論が復帰した後だけ通常制御へ戻ります。

## モデル配置

export済み`policy_torch.pt`と`policy_manifest.json`を`models/`へ配置します。形式は[`models/README.md`](models/README.md)を参照してください。weightは現時点では同梱していません。

リポジトリルートのinstall targetは、export bundleのSHA-256をManifestと照合してから所定位置へ配置します。

```bash
make lidar-rl-install-policy \
  LIDAR_RL_BUNDLE=aichallenge/ml_workspace/lidar_racing_rl/exported/<run>
```

## 起動

```bash
ros2 launch lidar_racing_controller lidar_racing_controller.launch.xml
```

提出スタックでは`aichallenge_submit_launch/reference.launch.xml`の`control_method:=lidar_racing`から起動します。

リポジトリルートからAWSIM開発環境を起動する場合は、LiDARをCPU modeで有効化し、制御方式も設定する専用ターゲットを使います。通常の`make dev`はLiDARを無効化するため、このコントローラの確認には使用しません。

```bash
make lidar-rl-awsim
make lidar-rl-request-control
```

後者はROS graphの起動後に実行し、AWSIMへ明示的に制御許可を送ります。4台時は`make lidar-rl-awsim4`の後に`make lidar-rl-request-control4`を使います。

sealed評価イメージでは、提出物を封入したイメージを再ビルドしてから次のように起動します。`CONTROL_METHOD=lidar_racing`の場合、未指定の`LIDAR_MODE`はCPUへ自動選択されます。

```bash
./create_submit_file.bash
./docker_build.sh eval --submit submit/aichallenge_submit.tar.gz
CONTROL_METHOD=lidar_racing make eval
```

既定の`debug.enabled: true`では、設定した集計間隔ごとにActor推論レイテンシのp50・p95・最大値・sample数をログへ出力します。不要な場合だけfalseへ変更できます。

## テスト

```bash
colcon test --packages-select lidar_racing_controller
colcon test-result --verbose
```

PyTorch、NumPy、ROS 2を含む実行テストはAI Challenge Docker内で行います。
