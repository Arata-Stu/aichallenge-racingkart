# Third-Party Notices

このサブプロジェクトが調査、直接import、または再配布する第三者コードを記録します。読み取り調査に使ったupstream SHA、実際に依存するfork submodule SHA、transitive依存の固定SHAを区別します。`PENDING`の項目を残したままリリースや再配布を行わないでください。

## F1TENTH Gym JAX

| 項目 | 内容 |
|---|---|
| リポジトリ | F1TENTH Gym JAX |
| upstream URL | `https://github.com/f1tenth/f1tenth_gym_jax.git` |
| fork URL | `https://github.com/Arata-Stu/f1tenth_gym_jax.git` |
| 読み取り調査upstream SHA | `1b4eb3f5161756bb925987753b965b549097742f` |
| 登録済みfork基点SHA | `1b4eb3f5161756bb925987753b965b549097742f` |
| fork変更後の固定SHA | `PENDING_FORK_CHANGE_COMMIT` |
| fork作業ブランチ | `aichallenge/dynamic-lidar` |
| ライセンス | MIT License |
| 使用箇所 | `repos/f1tenth_gym_jax`（editable Git submodule） |
| fork変更 | all-agent array step/reset、外部LiDAR/corruption hook、scan-only観測、終了種別分離、左右コース境界API。AI Challenge固有のOBB・報酬・NPC・SACは親側に保持する。 |

登録済み基点SHAには現在の未commit fork差分が含まれません。fork差分をreview・commit・user forkへpushした後の完全SHAを`fork変更後の固定SHA`と親gitlinkへ記録するまで、変更済みforkを固定したとは扱いません。submodule内の`LICENSE`にある著作権表示は次のとおりです。

```text
Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng
Copyright (c) 2023 Hongrui Zheng, Renukanandan Tumu, Luigi Berducci, Ahmad Amine
```

MIT Licenseの許諾・免責全文はsubmodule内の原文`LICENSE`に保持されています。再配布物にも同原文を含め、変更後SHAが`PENDING`のまま成果物を配布しないでください。

## jax-pf

| 項目 | 内容 |
|---|---|
| リポジトリ | jax-pf |
| upstream URL | `https://github.com/hzheng40/jax_pf` |
| 固定コミットSHA | `1b1417d7d2afbf24a9c6594195c2e872a6b4460a` |
| ライセンス | MIT License（GitHub primary repository表示） |
| 使用箇所 | `f1tenth` extraのray marcher依存。`pyproject.toml`の`tool.uv.sources`で直接固定。 |
| 固定根拠 | F1TENTH Gym JAX upstream `uv.lock`の依存監査 |
| JAX互換性 | package metadataの古い`jax<0.7`上限を、F1TENTH forkと親プロジェクトのuv overrideで`jax>=0.7.2,<0.8`へ統一。fork lockfileでは固定SHAとJAX 0.7.2の組合せを使用。親のCUDA extraはoverrideによるextra消失を避けるため`jax-cuda12-plugin[with-cuda]`も直接要求する。 |

正確なcopyright本文は未取得です。名称や著作者を推測して記載せず、依存を配布する前に固定SHAの原文licenseを取得して、MIT Licenseの著作権表示・許諾・免責全文を本noticeまたは配布物へ反映してください。

## 参考リポジトリ（コード非同梱）

次のリポジトリは実装調査用であり、このサブプロジェクトから直接importまたは再配布しません。取得する場合は[`repos/reference.repos`](repos/reference.repos)のプレースホルダーを完全なコミットSHAへ置換します。

| リポジトリ | URL | 固定コミットSHA | 用途 |
|---|---|---|---|
| Stoix | `https://github.com/EdanToledo/Stoix.git` | `PENDING` | JAX強化学習実装の参照 |
| End2Race | `https://github.com/michigan-traffic-lab/End2Race.git` | `PENDING` | End-to-End racing設計の参照 |
| f1tenth_development_gym | `https://github.com/F1Tenth-INI/f1tenth_development_gym.git` | `PENDING` | F1TENTH環境実装の参照 |

親リポジトリ自体はリポジトリルートの[LICENSE](../../../LICENSE)に従います。
