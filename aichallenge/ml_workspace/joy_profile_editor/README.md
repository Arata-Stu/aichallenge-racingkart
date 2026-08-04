# AIC Joy Profile Editor

Small browser UI for generating `teleop_manager` parameter YAML from actual
Linux joystick input (`/dev/input/js0`).

It is intentionally outside ROS packages to avoid merge conflicts with the
upstream repository. It writes this file by default:

```text
aichallenge/workspace/src/aichallenge_tools/teleop_manager/config/teleop.param.yaml
```

## Start

Run this inside the development container or on a Linux host that can read
`/dev/input/js0`:

```bash
cd /aichallenge/ml_workspace/joy_profile_editor
bash scripts/start_joy_profile_editor.sh
```

Open:

```text
http://127.0.0.1:8767/
```

## Notes

The UI shows both `/dev/input/js0` values and Browser Gamepad API values. Use
the `js0` capture result for ROS parameters, because browser axis/button indexes
can differ from the Linux joystick indexes used by `joy_node`.

The current AIC `teleop_manager_node` treats `speed_axis_index` as a signed
axis. If R2/L2 are used as throttle/reverse triggers, their released value may
not map safely without changing the teleop node. Start with the left stick
vertical axis for speed unless the teleop node is extended for trigger
normalization.
