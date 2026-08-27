# AWSIM calibration assets

このディレクトリには、AWSIM実測値から得たLiDAR統計と車両同定結果を置きます。
推測値を学習設定へ転記せず、使用したrosbag、計測条件、生成日時を成果物とともに
記録してください。

## LiDAR rosbag解析

解析スクリプトはROS 2の `rosbag2_py` とLaserScan型サポートを使うため、学習専用
Dockerではなく、既存のAI Challenge devコンテナ内で実行します。リポジトリルートの
`output/` はコンテナ内の `/output`、`aichallenge/` は `/aichallenge` にマウントされます。

まず開発imageを用意します。

```bash
./docker_build.sh dev
```

MCAPまたはSQLiteのrosbag2ディレクトリを指定して解析します。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_bag.py /output/20260827/d1/rosbag2_all' \
docker compose run --rm --no-deps autoware-command
```

単一の `.mcap` / `.db3` ファイルも指定できますが、分割bagや圧縮bagでは
`metadata.yaml` を含むrosbag2ディレクトリを指定してください。

保存形式を自動判別できない場合は明示します。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_bag.py /output/20260827/d1/rosbag2_all --storage-id mcap' \
docker compose run --rm --no-deps autoware-command
```

出力は既定で次の2ファイルです。同名ファイルは原子的に置換されます。

```text
assets/calibration/lidar_statistics.json
assets/calibration/lidar_statistics.md
```

短い読み取り確認には `--max-frames`、出力を分ける場合は `--output-dir` を使います。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_bag.py /output/20260827/d1/rosbag2_all --max-frames 1000 --output-dir /output/lidar-check' \
docker compose run --rm --no-deps autoware-command
```

## 集計内容

- beam数と `frame_id`
- `angle_min`、`angle_max`、`angle_increment`
- `range_min`、`range_max`、`scan_time`、`time_increment`
- bag timestamp／header timestampによるPublish周期
- valid、NaN、正負Inf、範囲外sampleの件数と比率
- 連続欠損runの幅と分布
- beam index／角度別の欠損率
- 距離別欠損率の代理推定
- 同一連続frameによるframe-hold候補

距離別欠損率では、欠損sampleの真の距離は観測できないため、同じbeam indexで最後に
得られた有効距離を代理値として使います。JSONにも
`classification: proxy_not_ground_truth` と手法を保存します。
既定の1.0 m bin幅はレポートの集計解像度であり、学習用noise値ではありません。
必要なら `--distance-bin-width` で変更し、実行条件として記録してください。

frame-holdは既定でrange列の完全一致を候補とします。量子化誤差を許容する必要があり、
かつ実測根拠がある場合だけ `--frame-hold-atol` を明示してください。一致は候補であり、
センサ停止や通信障害の確定判定ではありません。

## 壁抜けの扱い

LaserScanだけでは「壁を抜けた遠距離値」と「実際に開けた方向」を区別できません。
そのため、出力は常に次の分類を明記し、壁抜けの確定件数・頻度を `null` にします。

```text
not_determinable_without_gt_or_reference_wall_distance
```

時間方向の急な遠距離変化を調べる場合のみ、根拠のある閾値を明示できます。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_bag.py /output/20260827/d1/rosbag2_all --far-jump-threshold 5.0' \
docker compose run --rm --no-deps autoware-command
```

この結果も `temporal_far_jump_only_not_wall_leak` というヒューリスティック候補であり、
壁抜けとは断定しません。確定には、各frameへ時刻同期した地図／GT ray-cast壁距離が
必要です。

## 校正値として採用する前の確認

1. 計測シナリオ、AWSIM版、車両、コース、rosbagの所在を記録する。
2. beam数・角度metadata・range boundsが全frameで安定していることを確認する。
3. bag timestampとheader timestampの周期差を確認する。
4. MarkdownのwarningとJSONの分類・代理推定を確認する。
5. 複数シナリオで再計測し、代表性を確認してからdomain randomizationへ反映する。

生成済みJSONを手編集せず、解析条件を変える場合はコマンドと元rosbagを記録して
再生成してください。

## 車両応答の計測契約

`scripts/analyze_awsim_vehicle_response.py` は、bag全体を一度だけ順次読み取り、明示的に
指定した実験区間だけから車両応答を同定します。区間、励起量、許容値は自動推測しません。
topic、型、時刻、必要sampleのいずれかが不足すると、そのparameterは理由付き `null` の
ままです。7値の一つでも同定できない、5種類の実験が揃わない、またはsine sweep検証区間が
無効な場合は、YAMLを保存した後に終了code 3を返します。

repo内のAWSIM通信実装とlaunchから確認した既定契約は次のとおりです。

| 用途 | topic | message型 | 使用field |
|---|---|---|---|
| 入力 | `/control/command/control_cmd` | `autoware_auto_control_msgs/msg/AckermannControlCommand` | `lateral.steering_tire_angle`, `longitudinal.acceleration` |
| 実操舵 | `/vehicle/status/steering_status` | `autoware_auto_vehicle_msgs/msg/SteeringReport` | `steering_tire_angle` |
| 速度・yaw rate | `/vehicle/status/velocity_status` | `autoware_auto_vehicle_msgs/msg/VelocityReport` | `longitudinal_velocity`, `heading_rate` |
| 代替運動状態 | `/localization/kinematic_state` | `nav_msgs/msg/Odometry` | `twist.twist.linear.x`, `twist.twist.angular.z` |
| 代替加速度 | `/localization/acceleration` | `geometry_msgs/msg/AccelWithCovarianceStamped` | `accel.accel.linear.x` |

AWSIM Unity側のcommand subscriber実装はこのrepoに含まれないため、`longitudinal.speed`と
`longitudinal.acceleration`を同時に設定した場合の内部優先順位までは静的に断定できません。
最初の専用bagで、送信command、velocity status、実操舵statusが同じ実験に対応していることを
確認してください。解析scriptは観測されていない内部応答を補完しません。

既定ではmessage stampを使用します。制御指令はnested stampから外側のstamp、statusは
status stampからheader stampの順で探します。正のstampが無いsampleはbag timestampへ
黙って置換せず除外します。stamp逆行も該当seriesを推定不能にします。受信時刻を使った
end-to-end計測が目的の場合だけ `--timestamp-source bag` を明示し、manifestの
`time_reference` を `bag_start_seconds` へ変更してください。

既定の加速度応答は `/vehicle/status/velocity_status` の前後sampleから得る中心差分です。
不等間隔timestamp用の3点式を使い、manifestの `maximum_derivative_gap_seconds` を超える
sample間隔や、明示した加速度step区間の外側にあるsampleは使用しません。
`/localization/acceleration` は `twist2accel` のlow-pass処理を含むため、
`--acceleration-response localization_acceleration` を明示した場合だけ使用します。この
場合のgain・delayはraw AWSIM車両だけでなく、localization pipelineを含むend-to-end値です。

## 車両応答bagの収録

既存の計測用bag設定だけでは実操舵statusが入らない場合があります。AI Challenge dev
コンテナで、少なくとも次のtopicを同じbagへ収録してください。

```bash
ros2 bag record -s mcap \
  /control/command/control_cmd \
  /vehicle/status/steering_status \
  /vehicle/status/velocity_status \
  /localization/kinematic_state \
  /localization/acceleration \
  /clock
```

安全を確認した専用シナリオで、操舵step、操舵sine sweep、加速度step、加速後coast、
一定速度旋回を分離して実施します。入力振幅、時間、速度、閾値は車両・コース・AWSIM版の
実測条件なので、このrepoでは既定値を置きません。飽和や衝突がある区間は採用しないで
ください。

## 車両応答の解析

テンプレートをコピーし、すべての `null` をbagで確認した区間・受入値に置き換えます。
時刻は既定では、選択されたtopic群で得られた正のmessage stampの最小値を0秒とします。
reset等で各seriesのstampが逆行したbagは、そのseriesを使う推定を拒否します。

```bash
cp assets/calibration/awsim_vehicle_experiments.template.json \
  /output/awsim_vehicle_experiments.json
```

devコンテナ内で解析します。

```bash
CMD_WORKDIR=/aichallenge/ml_workspace/lidar_racing_rl \
CMD='python3 scripts/analyze_awsim_vehicle_response.py \
  /output/vehicle-response-bag \
  --experiments /output/awsim_vehicle_experiments.json' \
docker compose run --rm --no-deps autoware-command
```

既定出力は `assets/calibration/awsim_vehicle_model.yaml` です。最初に置かれている同名fileは
全値 `null` の未計測placeholderであり、学習用の値ではありません。

推定法は次のとおりです。

- 操舵gain・時定数・delay: 指令と実操舵のstep応答をFOPDTの10%／63.2% crossingで同定
- 加速度gain・delay: 加速度指令と速度差分、または明示したlocalization加速度のstep応答
- effective wheelbase: 一定旋回中の `v * tan(actual_steering) / yaw_rate` の中央値
- velocity drag: coast区間の `log(v)` と時間の最小二乗直線の負の傾き。
  manifestで明示した `minimum_r_squared` 未満のfitは棄却

複数の有効区間がある場合は候補の中央値を採用し、各候補・棄却理由・分散を同じYAMLへ
残します。操舵sine sweepは周波数区間を推測せず、step同定の検証用coverageとしてのみ
記録します。candidate間の差、coast回帰の `r_squared`、timestamp sourceを確認してから、
手動レビューを経て学習設定へ転記してください。
