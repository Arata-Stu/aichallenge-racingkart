# Virtual Scan RL

公式の`ml_workspace/reinforcement_learning`を変更せずに作った、AWSIM向けの
Virtual Scan強化学習パッケージです。アルゴリズムはStable-Baselines3のSAC、
観測エンコーダはTinyLiDARNetと同系統の5層1D CNNです。

## 現在の学習段階

まずはPlayer 1だけで1周を安定して完走する`lap`段階を実装しています。

- 観測: 障害物込みVirtual Scanの直近4フレームと、自車速度・yaw rate・操舵角・前回行動
- 行動: `[steering, longitudinal]`をそれぞれ`[-1, 1]`でSACが出力
- 縦制御: 正値は加速、負値はブレーキ。固定アクセルやルールベース制御は不使用
- 報酬: レースライン上の前進、セクション通過、完走、滑らかな操作、安全余裕
- 終了: 1周完了、衝突、コース逸脱、スタック、時間切れ

レースラインへの投影は前回セグメントの近傍だけを探索します。そのため、壁一枚で
隣接するヘアピンの反対側へ進捗が飛ぶことを防ぎます。

`configs/overtake.template.yaml`は次段階の設定置き場です。追い越し成功を正しく
報酬化するには、NPCシナリオのランダム化と、学習時だけ使う順位・passイベントの
生成が必要なので、現時点では追い越し学習用として実行しないでください。最終Policyへ
raw V2X座標を入力する必要はなく、入力は障害物込みVirtual Scanのままにできます。

## 起動

Autoware側は制御方式`virtual_scan_rl`で起動します。この方式はVirtual Scan生成と
Domain 0へのAWSIM Reset中継だけを起動し、別の制御ノードは立ち上げません。

```bash
# Autoware用Terminator内（Player 1）
# aic_player_menuで 5) Virtual Scan RL を選択

# 別ペイン。番号メニューで新規学習・再開・評価を選択
export ROS_DOMAIN_ID=1
cd /aichallenge/ml_workspace/virtual_scan_rl
bash run.sh
```

専用Terminatorを使う場合は、`make autoware-bash`でDockerへ入ったあとに次を実行します。

```bash
bash /aichallenge/utils/run_virtual_scan_rl_terminator.bash
```

画面内で`(1) Autoware`、`(2) AWSIM`、`(3) SAC runner`の順にEnterを押します。
学習・再開・評価、チェックポイント、Joy介入の有無はSAC runnerを実行した時点で
番号選択します。右側にはGPUとチェックポイント/介入データの状態が表示され、
`Stop All`はAutoware、AWSIM、RL runner（配下のjoy_nodeとlearnerを含む）を停止します。

RL実行中は`teleop_manager_node`を同時に起動しないでください。双方が
`/control/command/control_cmd`へpublishして制御が競合します。`run.sh`は必要なら
`joy_node`だけを子プロセスとして起動し、終了時にPIDを指定して停止します。

エピソード終了時は、SAC runnerペインに`reason`、step数、進捗、速度、最小障害物距離、
レースラインからの距離を表示します。開始直後のランダムPolicyが動き出す時間を確保するため、
停止によるResetは既定で10秒の猶予後、さらに10秒間停止した場合に行います。

AWSIMの`lapCount`は最初のスタートライン通過でも更新されるため、最初の変化は計測開始、
次の変化を1周完了として扱います。これとローカル投影したraceline約1周分の累積進捗を
併用し、どちらかが1周を確定した場合にエピソードを終了します。時間上限は50 Hz換算で
約300秒です。

Reset後は固定時間sleepしません。Domain 0の`/admin/awsim/state`を
Domain 1の`/awsim/admin_state`へブリッジします。Reset後の準備・カウントダウンを経た
管理状態`Start`と、車両固有の`/awsim/state`が`Ready`または`Start`になったことを
確認してからstep 0を開始します。待機中の時間は報酬、stuck判定、学習stepに含みません。

## Joy介入

既定ではDualShockのbutton index 2を押している間だけ、人間のステアとR2/L2を採用します。
離すとSACへ戻ります。介入時は次の2点を同時に行います。

1. `info.executed_action`を通じて、実際に車両へ送った人間行動をReplay Bufferへ保存
2. Scan、state、AI提案、人間行動、報酬を`interventions/*.npz`へ保存

これにより「AI提案を保存したのに車両は人間操作で動いた」という遷移の不整合を防ぎます。
介入データは次段階で補助的なBehavior Cloning lossにも利用できます。

## 出力

チェックポイント、Replay Buffer、TensorBoardログ、介入データはすべて`.gitignore`対象です。
設定値は[configs/lap.yaml](configs/lap.yaml)を編集すればよく、Pythonコードの変更や
再ビルドは不要です。
