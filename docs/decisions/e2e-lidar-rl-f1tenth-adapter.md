# ADR: F1TENTH Gym JAXとの統合境界

- 状態: 採用（fork変更のcommit SHA固定待ち）
- 日付: 2026-08-27

## コンテキスト

動的車両LiDAR、外部初期配置、Ego単位の終了処理は、変更前のF1TENTH Gym JAX公開APIだけでは完結しなかった。`Arata-Stu/f1tenth_gym_jax`のforkをsubmoduleとして登録し、AI Challenge固有でない配列APIとLiDAR hookをfork側へ実装する。公式upstreamへ変更を加えることは許可されない。

API調査とfork変更の基点はupstreamの`1b4eb3f5161756bb925987753b965b549097742f`である。現在のsubmodule作業ブランチ`aichallenge/dynamic-lidar`には未commit差分があるため、この基点SHAは変更後の固定SHAを意味しない。

## 決定

1. `repos/f1tenth_gym_jax`はuser forkだけをpush先とし、公式upstreamのpush URLを`DISABLED`にする。
2. forkにはall-agent array action/observation、scan-only観測、外部scan/corruption hook、外部pose/state reset、`terminated`/`truncated`分離、コース境界幅APIだけを置く。
3. 親リポジトリ側には動的OBB scan、AI Challenge固有の報酬、NPC、Ego単位のauto-reset、SAC学習を残す。
4. 親側はforkの公開`reset_array`、`reset_from_frenet_poses`、`step_env_array`だけを利用し、private `_scan`や辞書観測へ依存しない。
5. `pyproject.toml`は浮動Git参照ではなく、登録済みsubmoduleのeditable pathだけを参照する。
6. fork変更をreview・commit・forkへpushした完全SHAを親gitlinkとnoticeへ固定するまで、成果物をリリースしない。
7. 依存を実行していない間は、rollout、JIT、benchmark、学習の成功を主張しない。

## forkに実装した汎用変更

- map scanへ動的障害物scanを合成する公開hook
- 外部から初期Stateを指定するreset hook
- scan corruption hook
- `terminated`と`truncated`の分離
- LiDAR-only観測モード
- 辞書・agent Python反復を介さないarray action / LiDAR-only step hook

AI Challenge固有の報酬、NPC設定、ROSトピック、学習ループは親リポジトリ側に残す。

## 影響

- forkがなくても親側の純粋関数とテストを先行できる。
- source上のprivate API依存とvehicle-axis辞書変換は学習経路から除去された。
- 依存環境でのJIT/integration testは未実行であり、fork commit SHA固定後に必須となる。
- 公式upstreamは取得専用であり、push先として使用しない。
