# rl_train_controller

このディレクトリは、ROS 2のresetトピックをPlayerドメインからAWSIMドメインへ、
AWSIMの管理状態を逆方向へ中継するパッケージです。Joycon起動時にも自動的に
起動されるため、物理Joyデバイスを2つの`joy_node`から開く必要はありません。

- `ROS_DOMAIN_ID`（通常はJoycon PlayerのDomain 1）の`/awsim/reset`を購読
- `AWSIM_DOMAIN_ID`（デフォルト0）の`/admin/awsim/reset`に転送
- Domain 0の`/admin/awsim/state`を購読
- Player Domainの`/awsim/admin_state`へ転送


## ディレクトリ構成

```text
rl_train/
  CMakeLists.txt
  package.xml
  launch/
    rl_train.launch.xml
  rl_train_controller/
    __init__.py
    rl_train_controller_node.py
```

## ノード概要

実装: `rl_train_controller/rl_train_controller_node.py`

- ノード名: `awsim_reset_domain_bridge`
- Reset購読 (parameter): `src_topic` (default: `/awsim/reset`)
- Reset配信 (parameter): `dst_topic` (default: `/admin/awsim/reset`)
- 管理状態購読 (parameter): `admin_state_topic` (default: `/admin/awsim/state`)
- 管理状態配信 (parameter): `forwarded_state_topic` (default: `/awsim/admin_state`)
- メッセージ型: Resetは`std_msgs/msg/Empty`、管理状態は`std_msgs/msg/String`

内部でROS 2 Contextを2つ作成し、Player DomainとAWSIM Domainを同時に扱います。

Joyconでは`teleop.param.yaml`の`reset_button_index`を押すと、Player Domainへ
`/awsim/reset`が1回だけpublishされます。現在のDualShock 4設定はbutton index 0
（×ボタン）です。中継ノードがそれをDomain 0の`/admin/awsim/reset`へ転送します。
