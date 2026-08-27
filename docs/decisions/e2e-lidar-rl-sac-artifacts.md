# ADR: SAC Replay・checkpoint・モデルexport境界

- 状態: 採用（依存環境での実行検証待ち）
- 日付: 2026-08-28

## コンテキスト

LiDAR観測は`[4, 2, 360]`であり、現在観測と次観測をfloat32のまま大量保存するとReplay Bufferが大きくなる。また、学習途中のcheckpoint、評価用Actor、ROS 2提出用PyTorch modelは用途と信頼境界が異なる。pickleでPython module全体を保存したり、GT状態をReplayへ混入させたりすると、再現性とLiDAR-only制約を保証できない。

## 決定

1. Replay Bufferは固定shapeのJAX ringとし、保存schemaを`observation`、正規化action、reward、`terminated`、`truncated`、`next_observation`に限定する。GT pose、Frenet状態、NPC transitionは型として保持しない。
2. LiDAR観測はReplay内だけfloat16で保持し、sample時にfloat32へ戻す。既定capacityは100,000で、現在・次観測を含む割当量は約1.07 GiBとする。
3. time-limitの`truncated`はCritic targetのbootstrapを残し、真の`terminated`だけmaskする。
4. learner checkpointはFlax msgpackで、Actor単体payloadとmetadataを併記する。metadataにはarchitecture、resolved config hash、root/fork commit、payload checksumを含める。
5. checkpoint schema v2はlearner stepに加えて累積environment transitionを保存する。checkpoint directoryは一時directoryへ完全に書いた後にrenameし、既存の番号付きcheckpointを上書きしない。同一payloadの再保存だけは実ファイルchecksumを再検証して冪等回復する。`LATEST`更新までcheckpoint rootのadvisory lockで直列化し、新しいstepから古いstepへ巻き戻さない。
6. 通常checkpointへReplay Bufferと環境状態を含めない。resumeはoptimizer/Actor/Critic/temperatureと累積進捗を復元したwarm restartであり、新しいReplayを再収集してから更新を再開する。bit-exactなrollout再開とは扱わない。
7. 学習開始時にrootとforkが未commitならfail closedする。resolved config、commit SHA、status、dirty diff、環境情報をrun directoryへ保存する。
8. 評価はcheckpointのconfig hashとresolved configを照合し、Actorの`tanh(mean)`だけを使用する。
9. exportはActor単体msgpackを読み、同形PyTorch Actorへparameterを明示変換する。ランダムcanonical入力で最大絶対誤差`1e-5`以下を確認できた場合だけ、weights-only state dictとManifestを発行する。
10. ROS 2パッケージへの配置時にもManifestのmodel checksumを再検証し、modelを先、Manifestを後に置換する。

## 影響

- Replay schema自体がActor/CriticへのGT混入を防ぐ。
- checkpointはReplay全体を含まないため小さく移植しやすいが、resume直後はwarmup transitionの再収集が必要になる。
- root/forkの未commit差分を残したまま本学習を開始できない。実装reviewとfork SHA固定が先行条件になる。
- checkpoint、決定論評価、Flax/PyTorch parity、ROS runtimeの実動作は、Ubuntuの固定依存環境で別途検証する必要がある。
