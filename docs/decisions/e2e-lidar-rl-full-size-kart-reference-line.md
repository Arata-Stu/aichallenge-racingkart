# ADR: フルサイズカートの教師・NPC基準線

- 状態: 採用（依存環境での走行検証待ち）
- 日付: 2026-08-28

## コンテキスト

F1TENTH Gym JAXの例はSpielbergの最適化racelineをPure Pursuitへ渡す。一方、AI Challenge車両設定の幅は1.45 mで、Spielberg centerline CSVの左右半幅は最小1.1 mである。車体半幅0.725 mを引くとセンターから使える静的な横余裕は片側0.375 mしかない。調査した既定racelineは少なくとも一部でcenterlineから約0.85 m離れており、小型F1TENTH車両には成立しても本車両の基準線にはできない。

## 決定

1. 初期の教師方策とPure Pursuit NPCは`track.centerline`だけを基準線にする。既定racelineへの暗黙fallbackを設けない。
2. centerline waypointの速度は設定の`base_target_speed`から明示的に生成する。教師とNPCで異なる基準速度を許可しない。
3. NPCの横offsetは初期値を`[-0.2, 0.2]` mに制限する。実行開始時にtrack左右幅から車体半幅を引き、設定範囲が静的余裕内に収まることをfail closedで検証する。
4. reset時は位置jitterと車体の回転投影を含む境界余裕を別途検証する。
5. これらは必要条件であり、操舵遅れやheading errorを含む動的安全性の保証ではない。Ubuntu上の単独・4台rolloutでoff-trackと接触を測定してから速度・offsetを校正する。
6. ActorとCriticは引き続きLiDAR-onlyとし、centerlineやGT poseは教師・NPC・報酬・評価の許可された境界にだけ置く。

## 影響

- 小型車向けracelineを流用して初期状態から車体が境界外へ出る失敗を防ぐ。
- 追い抜きに必要な横方向の多様性は狭くなる。Step 2の実走結果とコース別幅を確認後、安全域内で設定を調整する必要がある。
- centerline APIと左右幅はforkの固定SHAに含まれるため、submodule更新時に契約testが必要になる。
