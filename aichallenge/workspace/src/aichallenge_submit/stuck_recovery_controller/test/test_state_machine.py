from stuck_recovery_controller.state_machine import (
    RecoveryConfig,
    RecoveryState,
    RecoveryStateMachine,
    is_stuck_candidate,
)


def update(machine, now, *, stuck=False, velocity=0.0, position=(0.0, 0.0), gear=2,
           manual=False):
    return machine.update(
        now,
        stuck_candidate=stuck,
        velocity=velocity,
        position=position,
        gear=gear,
        manual_trigger=manual,
    )


def test_complete_recovery_cycle():
    config = RecoveryConfig(
        startup_grace_period=0.0,
        stuck_timeout=1.0,
        stop_hold_duration=0.2,
        gear_settle_duration=0.1,
        reverse_duration=2.0,
        reverse_distance=1.0,
        cooldown_duration=3.0,
    )
    machine = RecoveryStateMachine(config)

    update(machine, 0.0, stuck=True)
    update(machine, 1.1, stuck=True)
    assert machine.state == RecoveryState.STOPPING

    update(machine, 1.2)
    update(machine, 1.5)
    assert machine.state == RecoveryState.SHIFT_REVERSE

    update(machine, 1.7, gear=20)
    assert machine.state == RecoveryState.REVERSING

    update(machine, 2.0, position=(1.1, 0.0), gear=20)
    assert machine.state == RecoveryState.STOPPING_REVERSE

    update(machine, 2.1, gear=20)
    update(machine, 2.4, gear=20)
    assert machine.state == RecoveryState.SHIFT_DRIVE

    update(machine, 2.6, gear=2)
    assert machine.state == RecoveryState.NORMAL
    assert machine.cooldown_until == 5.6


def test_stuck_timer_resets_when_vehicle_moves():
    config = RecoveryConfig(startup_grace_period=0.0, stuck_timeout=1.0)
    machine = RecoveryStateMachine(config)

    update(machine, 0.0, stuck=True)
    update(machine, 0.7, stuck=False, velocity=1.0)
    update(machine, 1.2, stuck=True)
    update(machine, 1.8, stuck=True)

    assert machine.state == RecoveryState.NORMAL


def test_manual_trigger_bypasses_grace_period():
    machine = RecoveryStateMachine(RecoveryConfig(startup_grace_period=30.0))

    update(machine, 10.0, manual=True)

    assert machine.state == RecoveryState.STOPPING


def test_missing_gear_feedback_uses_timeout():
    config = RecoveryConfig(
        startup_grace_period=0.0,
        stop_hold_duration=0.0,
        gear_feedback_timeout=0.5,
    )
    machine = RecoveryStateMachine(config)

    update(machine, 0.0, manual=True)
    update(machine, 0.1)
    assert machine.state == RecoveryState.SHIFT_REVERSE
    update(machine, 0.7, gear=None)
    assert machine.state == RecoveryState.REVERSING


def test_explicit_wrong_gear_does_not_start_reverse():
    config = RecoveryConfig(
        startup_grace_period=0.0,
        stop_hold_duration=0.0,
        gear_feedback_timeout=0.1,
    )
    machine = RecoveryStateMachine(config)

    update(machine, 0.0, manual=True)
    update(machine, 0.1)
    update(machine, 1.0, gear=2)

    assert machine.state == RecoveryState.SHIFT_REVERSE


def test_forward_command_and_zero_velocity_is_stuck_candidate():
    assert is_stuck_candidate(
        velocity=0.05,
        velocity_is_fresh=True,
        command_speed=2.0,
        command_acceleration=0.7,
        command_is_fresh=True,
        mpc_infeasible=False,
        velocity_threshold=0.2,
        command_speed_threshold=1.0,
        command_acceleration_threshold=0.3,
    )


def test_infeasible_mpc_is_candidate_even_with_stop_command():
    assert is_stuck_candidate(
        velocity=0.0,
        velocity_is_fresh=True,
        command_speed=0.0,
        command_acceleration=-1.6,
        command_is_fresh=True,
        mpc_infeasible=True,
        velocity_threshold=0.2,
        command_speed_threshold=1.0,
        command_acceleration_threshold=0.3,
    )


def test_infeasible_mpc_does_not_trigger_while_vehicle_is_moving():
    assert not is_stuck_candidate(
        velocity=0.5,
        velocity_is_fresh=True,
        command_speed=0.0,
        command_acceleration=-1.6,
        command_is_fresh=True,
        mpc_infeasible=True,
        velocity_threshold=0.2,
        command_speed_threshold=1.0,
        command_acceleration_threshold=0.3,
    )


def test_stale_velocity_never_triggers_recovery():
    assert not is_stuck_candidate(
        velocity=0.0,
        velocity_is_fresh=False,
        command_speed=2.0,
        command_acceleration=0.7,
        command_is_fresh=True,
        mpc_infeasible=True,
        velocity_threshold=0.2,
        command_speed_threshold=1.0,
        command_acceleration_threshold=0.3,
    )
