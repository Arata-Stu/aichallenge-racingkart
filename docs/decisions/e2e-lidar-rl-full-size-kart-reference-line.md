# ADR: F1TENTH学習車両とAWSIM実車プロファイルの分離

- 状態: 採用（Ubuntu GPU走行結果を反映）
- 日付: 2026-08-28

## コンテキスト

F1TENTH Gym JAXのSpielbergは既定車両（全長0.58 m、全幅0.31 m、wheelbase 0.3302 m）向けである。AI Challenge車両（全長2.0 m、全幅1.45 m、wheelbase 1.087 m）を同じmapへ載せると、左右半幅1.1 mに対する静的余裕は片側0.375 mしかない。

Ubuntu GPU上の300-step診断では、実寸車両は最大横偏差0.243 m、最大heading誤差2.923 rad、最小車体境界margin -0.322 mとなった。1000-stepでは91回異常終了し、90回off-track、80回collisionだった。centerlineと走行軌跡のSVG確認ではcenterline点列に破損はなく、車両とコースの相対寸法が成立していないことを確認した。

## 決定

1. SpielbergでのStep 1/2初期学習は`f1tenth_nominal`車両プロファイルを使う。
2. AWSIM・ROS 2推論は`aichallenge_kart`とdeployment設定の実車action上限を使い、学習mapの車両寸法から分離する。
3. Actorは正規化actionを出力する。export manifestの実steering/acceleration上限は学習車両ではなくAWSIM deployment設定から生成する。
4. 初期の教師方策とPure Pursuit NPCは`track.centerline`だけを基準線にし、既定racelineへの暗黙fallbackを設けない。
5. AWSIM rosbagで車両応答を同定後、実車寸法・wheelbase・action応答をdomain randomizationまたはfine-tuneへ段階的に導入する。
6. ActorとCriticは引き続きLiDAR-onlyとし、車両poseやcenterlineは教師・NPC・報酬・評価の許可された境界にだけ置く。

## 影響

- F1TENTH内の学習成立性とAWSIM実車契約を混同しない。
- F1TENTHだけで得たcheckpointをsim2real完了済みとは扱わない。AWSIM評価とcalibrationを必須の後段gateにする。
- 学習とdeploymentでactionの物理上限が異なるため、正規化actionとmanifestの境界をtestで固定する。
