# TinyLiDARNet training Terminator

TinyLiDARNetの録画選択、前処理、GPU学習、学習状況確認を1画面で行うための専用レイアウトです。

## Start

ホスト側で実行します。

```bash
make tiny-lidar-training
```

| Pane | Purpose |
|---|---|
| Left | `run_pipeline.sh`による対話式の前処理・学習 |
| Top right | `nvidia-smi`（2秒更新） |
| Center right | 録画・train/val・checkpoint一覧（5秒更新） |
| Bottom right | `tiny_lidar_net_pytorch`学習workspaceの自由シェル |

左ペインには`./run_pipeline.sh`が入力済みです。Enterを押し、録画、train/val、学習条件を番号で
選択します。学習結果は`checkpoints/<日時>/best_model.pth`へ保存され、
`checkpoints/latest/best_model.pth`も更新されます。

録画はまとめて選択できます。例えば、録画1〜6と9をtrain、7と8をvalにする場合：

```text
Train recording numbers: 1-6 9
Validation recording numbers: 7 8
Existing outputs [skip/overwrite] (default: skip):
Run all selected preprocessing jobs? [Y/n]:
```

シーケンス名は自動生成され、選択した録画が順番に前処理されます。

右下シェルでは以下の短縮コマンドを利用できます。

```bash
tln-pipeline       # 対話式パイプライン
tln-status         # データセットとcheckpointの一覧
tln-checkpoints    # ROS 2走行時に選択できるcheckpointの一覧
```

各ペインの履歴は`/output/terminator-history/tiny-lidar-training`に保持されます。
