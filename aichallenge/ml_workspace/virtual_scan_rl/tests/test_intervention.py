import numpy as np

from virtual_scan_rl.intervention import joy_action, trigger_press


def test_dualshock_trigger_conversion():
    assert trigger_press(1.0) == 0.0
    assert trigger_press(-1.0) == 1.0


def test_joy_action_is_signed_longitudinal():
    axes = [0.25, 0.0, 1.0, 0.0, 0.0, -1.0]
    action = joy_action(
        axes, steer_axis=0, positive_axis=5, negative_axis=2, deadzone=0.05
    )
    np.testing.assert_allclose(action, [0.25, 1.0])
    axes[5], axes[2] = 1.0, -1.0
    action = joy_action(
        axes, steer_axis=0, positive_axis=5, negative_axis=2, deadzone=0.05
    )
    np.testing.assert_allclose(action, [0.25, -1.0])

