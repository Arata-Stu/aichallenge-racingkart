# External repositories

このディレクトリでは、外部リポジトリを用途別に管理します。

## `f1tenth_gym_jax/`

実際にimportし、汎用的な動的LiDAR機能を独自変更するforkのGit submoduleです。

- upstream: `https://github.com/f1tenth/f1tenth_gym_jax.git`
- fork: `https://github.com/Arata-Stu/f1tenth_gym_jax.git`
- fork側作業ブランチ: `aichallenge/dynamic-lidar`
- 配置先: `repos/f1tenth_gym_jax`

clone後の利用者は次を実行します。

```bash
git submodule update --init --recursive
git -C aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax remote -v
git -C aichallenge/ml_workspace/lidar_racing_rl/repos/f1tenth_gym_jax rev-parse HEAD
```

親リポジトリはsubmoduleの完全なコミットSHAを固定します。現在は変更前の基点SHA上で`aichallenge/dynamic-lidar`の差分をreview中です。fork変更をcommitしてuser forkへpushし、その新しいSHAへ親gitlinkを更新するまでは固定完了ではありません。submodule内でdetached HEADのままコミットしたり、upstreamへpushしたりしないでください。

公式upstreamはfetch専用で、push URLは`DISABLED`です。通常のpush先は`origin=https://github.com/Arata-Stu/f1tenth_gym_jax.git`に限定します。

## `reference.repos`

実装調査だけに使うリポジトリのvcstool形式一覧です。これらは直接importしないためsubmoduleにはしません。

安全のため、初期ファイルの`version`は意図的に無効なプレースホルダーです。すべてを40桁の完全なコミットSHAへ置換し、URL・ライセンス・用途を`THIRD_PARTY_NOTICES.md`へ記録するまで、`vcs import`を実行しないでください。`main`、`master`、`latest`などの浮動参照は禁止です。
