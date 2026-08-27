# ADR: LiDAR-only方策の情報境界

- 状態: 採用
- 日付: 2026-08-27

## コンテキスト

F1TENTH Gym JAXの公開観測は、`observe_others=false`でも自車のFrenet座標と車両状態を含む。一方、本プロジェクトのActorとCriticは、現在および過去の2D LiDAR rangeとvalidityだけを入力にしなければならない。GTは報酬、終了判定、初期配置、NPC、評価、およびシミュレータ内部のセンサ生成に限って利用できる。

## 決定

1. Actor・Criticへは上流の観測辞書を渡さない。
2. `LidarRacingEnv`はシミュレータの`State.scans`だけをcanonical形式へ変換し、Egoの`[frame_stack, 2, 360]`配列だけを公開する。
3. `State.cartesian_states`と`State.frenet_states`はラッパー内部の許可された処理、または固定NPC制御でのみ参照する。
4. Replay Bufferへ保存するtransitionはEgoだけとし、観測・行動・報酬・`terminated`・`truncated`・次観測以外のGTを保存しない。
5. Ego終了時はその並列環境全体をリセットする。他の並列環境は`vmap`内で継続させる。
6. 時間上限は`truncated`、衝突と完走は`terminated`として分離する。
7. Actorへ渡す型は辞書ではなく固定shapeの配列とし、GTフィールドを後から設定で追加できる汎用経路を設けない。

## 影響

- 上流の観測仕様が変わっても、方策入力は`State.scans`から組み立てる明示的な境界に留まる。
- 報酬・NPC・センサ生成のコードにはGT利用目的をコメントで残す必要がある。
- 上流State APIとscan生成hookの互換性は、submoduleの固定SHAとintegration testで管理する。
- 現行上流は壁接触と車両接触を単一の`collisions`へ集約する。評価時はcollision発生時のfiniteな車両OBB重なりを代理信号として相手接触を分類し、重なりがない場合だけ壁接触とする。不確定な非finite状態ではgeneric collisionだけを保持するため、厳密な接触原因とは扱わない。
- forkはcenterline CSVの左右幅を保持し、親wrapperは車体投影clearanceを含むoff-track判定を報酬と`terminated`へ接続する。
