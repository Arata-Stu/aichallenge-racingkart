# tiny_lidar_net_pytorch (training)

自車の2D LiDARだけを使うTinyLiDARNetのPyTorch学習用workspaceです。ROS 2推論コードは
`aichallenge/workspace/src/aichallenge_submit/tiny_lidar_net_pytorch`に分離しています。

学習結果はメタデータ付きのPyTorch `.pth`として保存され、ROS 2側から直接読み込めます。
重みを`.npy`や`.npz`へ変換する処理はありません。

## Dataset

既存の前処理結果を利用できます。各シーケンスディレクトリに以下の3ファイルを置きます。

- `scans.npy`: `[N, scan_points]`
- `accelerations.npy`: `[N]`
- `steers.npy`: `[N]`

Virtual Scanに合わせた既定入力は1080点です。保存されたscan点数が異なる場合は、学習時に1080点へ
線形補間します。

## Interactive preprocessing and training

通常はこちらを使用してください。Bag Managerの録画、train/val、学習パラメータを番号で選択できるため、
長いパスを入力する必要はありません。

ホストから学習専用Terminatorを起動する場合：

```bash
make tiny-lidar-training
```

学習用Terminatorは次の4ペインで構成されます。

- 前処理・学習パイプライン（Enterで`run_pipeline.sh`）
- `nvidia-smi` GPU監視
- データセット／チェックポイント一覧
- 学習workspaceを開いた自由シェル

## Web dashboard

録画選択、前処理、学習、前処理済みVirtual Scanの再生、ステア分布、ジョブログをブラウザで扱えます。
ホストから次を実行してください。

```bash
make tiny-lidar-dashboard
```

起動後に <http://localhost:8765> を開きます。終了は起動したターミナルで`Ctrl+C`です。
`make autoware-bash`のコンテナ内から起動する場合は次を使用します。

```bash
./ml_workspace/tiny_lidar_net_pytorch/run_dashboard.bash
```

ダッシュボードでは以下をパス入力なしで実行できます。

- R1/L1で停止した最新録画の自動検出と、カテゴリ・結果・採用可否・メモの保存
- 1つの録画への複数Dataset Versionタグ付け（Bag内の`collection_annotation.json`へ保存）
- Dataset Versionの作成・切替（例：`pretrain-v1`、`optimal-v1`、`recovery-v2`）
- Bag Manager録画を一括選択してtrain/valへ前処理（同じ録画の重複選択可）
- Virtual Scanのフレーム送り／自動再生とステア・加速度ラベルの確認
- ステア左右／直進比率、ヒストグラム、同期誤差、破棄Scan数の確認
- ステア専用の事前学習と、既存`.pth`からのFine-tuning
- 非同期ジョブの進捗・ログ確認と停止

既存の`datasets/train`と`datasets/val`は`default` Versionとしてそのまま認識されます。新しいVersionは
次のように完全に分離して保存されます。

```text
datasets/
├── train/                         # default
├── val/                           # default
└── versions/
    ├── pretrain-v1/
    │   ├── train/
    │   └── val/
    └── optimal-v1/
        ├── train/
        └── val/
```

画面上部の`ACTIVE DATASET VERSION`で、可視化対象と学習対象を同時に切り替えます。前処理画面では
既存Versionを選ぶか、新しいVersion名を入力します。名前付きVersionのチェックポイントは
`checkpoints/versions/<version>/<日時>/`へ保存され、`.pth`内部にも`dataset_version`が記録されます。

バックエンドPIDは`/output/tiny-lidar-dashboard/dashboard.pid`へ記録され、多重起動を防止します。
前処理・学習はジョブごとのPIDとプロセスグループで管理されます。画面からの停止やバックエンド終了時は
`SIGTERM`、猶予時間、必要な場合のみ`SIGKILL`の順で終了して`wait`するため、子プロセスを残しません。
バックエンドが強制終了した場合もLinuxのparent-death signalにより実行中ジョブを終了します。

前処理では録画を1件ずつ選ぶ必要はありません。録画一覧に対し、trainとvalをまとめて指定できます。

```text
Train recording numbers: 1-6 9 11
Validation recording numbers: 7 8 10
```

カンマ区切り（`1,2,5-8`）や`all`、`none`も利用できます。シーケンス名は録画パスから自動生成され、
全処理予定を一度確認してから連続前処理されます。

同じ録画番号をtrainとvalの両方に指定することもできます。同一コースで取得したデータや、同じ録画を
両方のsplitで確認したい場合もパイプライン側では制限しません。

Terminatorを使わず、コンテナ内の通常ターミナルで実行する場合：

```bash
cd /aichallenge/ml_workspace/tiny_lidar_net_pytorch
./run_pipeline.sh
```

既定では以下を自動的に使用します。

- 録画検索: `/aichallenge/record`
- データセット: `./datasets/train`、`./datasets/val`
- 学習結果: `./checkpoints/<日時>/best_model.pth`
- 最新モデル: `./checkpoints/latest/best_model.pth`

録画一覧や現在のデータセット、チェックポイントだけを確認する場合：

```bash
./run_pipeline.sh --list
```

別の保存場所を使う場合だけ、`--record-root`、`--dataset-root`、`--checkpoint-root`を指定します。

## Train

```bash
cd /aichallenge/ml_workspace/tiny_lidar_net_pytorch
python3 train.py \
  --train-dir /path/to/datasets/train \
  --val-dir /path/to/datasets/val \
  --output-dir ./checkpoints \
  --device cuda
```

主なオプションは次のとおりです。

```text
--architecture normal|small
--input-dim 1080
--batch-size 64
--epochs 100
--acceleration-weight 1.0
--steering-weight 1.0
--device auto|cpu|cuda|cuda:N
```

`best_model.pth`と`last_model.pth`には、モデル重みに加えてarchitecture、input_dim、max_rangeが
保存されます。ROS 2推論側はこれらを自動的に読み取ります。

## ROS 2 inference

```bash
export TINY_LIDAR_NET_PYTORCH_CHECKPOINT=\
/aichallenge/ml_workspace/tiny_lidar_net_pytorch/checkpoints/best_model.pth

ros2 launch aichallenge_submit_launch reference.launch.xml \
  simulation:=true use_sim_time:=true control_method:=tiny_lidar_net_pytorch
```

チェックポイントのパスを手入力したくない場合は、番号選択式の単体起動も利用できます。

```bash
/aichallenge/utils/run_tiny_lidar_player.bash 2
```

Player 2〜4を同時に使用する場合は、Terminatorの各Playerペインで`aic_player_menu`を実行し、
実行時に`tiny`とcheckpointを選択します。停止後にもう一度`aic_player_menu`を実行すれば、
Terminatorを開き直さず別のcheckpointへ変更できます。
