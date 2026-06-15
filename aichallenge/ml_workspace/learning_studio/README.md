# E2E Learning Studio

E2E model のデータ作成、学習、sensor frame 単位の評価を 1 つの
ローカル GUI で行います。

標準対応 model:

- `PilotNet`: camera image を入力
- `TinyLiDARNet`: 2D LiDAR scan を入力

## 起動

Docker container 内で次を実行します。

```bash
/aichallenge/ml_workspace/run_learning_studio.sh
```

ブラウザで `http://localhost:8765` を開きます。

record root や port を変更する場合:

```bash
E2E_RECORD_ROOT=/aichallenge/ml_workspace/rawdata \
E2E_STUDIO_PORT=9000 \
/aichallenge/ml_workspace/run_learning_studio.sh
```

## ディレクトリ

既定値は次の通りです。

| 用途 | Path |
| --- | --- |
| ROS bag 探索 root | `/aichallenge/record` |
| PilotNet dataset | `/aichallenge/ml_workspace/pilot_net/datasets` |
| TinyLiDARNet dataset | `/aichallenge/ml_workspace/tiny_lidar_net/datasets` |
| PilotNet 学習・評価結果 | `/aichallenge/ml_workspace/pilot_net/outputs/learning_studio` |
| TinyLiDARNet 学習・評価結果 | `/aichallenge/ml_workspace/tiny_lidar_net/outputs/learning_studio` |

環境変数で変更できます。

```bash
E2E_RECORD_ROOT=/path/to/record
E2E_PILOT_DATASETS_ROOT=/path/to/pilot/datasets
E2E_PILOT_OUTPUT_ROOT=/path/to/pilot/outputs
E2E_TINY_LIDAR_DATASETS_ROOT=/path/to/tiny_lidar/datasets
E2E_TINY_LIDAR_OUTPUT_ROOT=/path/to/tiny_lidar/outputs
E2E_STUDIO_PORT=8765
```

後方互換のため、`E2E_DATASETS_ROOT` と `E2E_STUDIO_OUTPUT_ROOT` は
PilotNet の保存先としても利用できます。

## Workflow

1. **Data**
   - record root 以下の `metadata.yaml` を再帰探索します。
   - `PilotNet` または `TinyLiDARNet` を選択します。
   - sequence ごとに `Train`、`Validation`、`Both`、`Unused` を選びます。
   - 選択 model に応じて image または LaserScan と control topic を抽出します。
2. **Train**
   - model ごとに作成した dataset と hyperparameter を選びます。
   - 各 workspace の既存 `train.py` を Hydra override 付きで実行します。
   - checkpoint は run ごとに保存されます。
3. **Evaluate**
   - dataset split と checkpoint を選択し、全 frame を推論します。
   - MAE の時系列と誤差上位 frame を表示します。
   - 再生、停止、前後移動、速度変更、target/prediction overlay を利用できます。
   - PilotNet は `conv4` と steering 出力を使った Grad-CAM を frame 単位で表示できます。
   - TinyLiDARNet の場合は scan をトップダウン表示します。

## Notes

- GUI はローカル開発用です。認証機能はないため、信頼できない network へ公開しないでください。
- 同時に実行できる extraction / training / evaluation job は 1 つです。
- dataset、run、evaluation は既存 directory を上書きしません。別名を指定してください。
- Stop job は実行中 subprocess group に `SIGTERM` を送ります。
