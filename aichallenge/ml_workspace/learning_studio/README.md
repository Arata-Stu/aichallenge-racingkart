# E2E Learning Studio

PilotNet のデータ作成、学習、画像単位の評価を 1 つのローカル GUI で行います。

## 起動

Docker container 内で次を実行します。

```bash
python3 /aichallenge/ml_workspace/learning_studio/server.py \
  --host 0.0.0.0 \
  --port 8765
```

ブラウザで `http://localhost:8765` を開きます。

## ディレクトリ

既定値は次の通りです。

| 用途 | Path |
| --- | --- |
| ROS bag 探索 root | `/aichallenge/record` |
| 抽出 dataset | `/aichallenge/ml_workspace/pilot_net/datasets` |
| 学習・評価結果 | `/aichallenge/ml_workspace/pilot_net/outputs/learning_studio` |

環境変数で変更できます。

```bash
E2E_RECORD_ROOT=/path/to/record
E2E_DATASETS_ROOT=/path/to/datasets
E2E_STUDIO_OUTPUT_ROOT=/path/to/outputs
E2E_STUDIO_PORT=8765
```

## Workflow

1. **Data**
   - record root 以下の `metadata.yaml` を再帰探索します。
   - sequence ごとに `Train`、`Validation`、`Both`、`Unused` を選びます。
   - image/control topic と画像前処理を設定し、既存
     `pilot_net/extract_data_from_bag.py` を実行します。
2. **Train**
   - 作成した dataset と hyperparameter を選びます。
   - 既存 `pilot_net/train.py` を Hydra override 付きで実行します。
   - checkpoint は run ごとに保存されます。
3. **Evaluate**
   - dataset split と checkpoint を選択し、全 frame を推論します。
   - MAE の時系列と誤差上位 frame を表示します。
   - 再生、停止、前後移動、速度変更、target/prediction overlay を利用できます。

## Notes

- GUI はローカル開発用です。認証機能はないため、信頼できない network へ公開しないでください。
- 同時に実行できる extraction / training / evaluation job は 1 つです。
- dataset、run、evaluation は既存 directory を上書きしません。別名を指定してください。
- Stop job は実行中 subprocess group に `SIGTERM` を送ります。
