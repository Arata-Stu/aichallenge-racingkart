import numpy as np

from virtual_scan_rl.reward import EpisodeTermination, LapReward, StepSignals


def test_lap_and_progress_are_positive_but_collision_dominates():
    config = {
        "progress_scale": 2.0,
        "section_bonus": 5.0,
        "lap_bonus": 200.0,
        "reverse_progress_scale": 4.0,
        "step_penalty": 0.02,
        "collision_penalty": 150.0,
        "off_track_penalty": 100.0,
        "steering_delta_penalty": 0.03,
        "acceleration_delta_penalty": 0.01,
        "clearance_threshold_m": 1.0,
        "clearance_penalty_scale": 0.1,
    }
    reward = LapReward(config)
    base = dict(
        progress_m=0.5,
        track_distance_m=0.0,
        lap_completed=False,
        section_changed=False,
        speed_mps=5.0,
        min_clearance_m=2.0,
        previous_action=np.zeros(2),
        action=np.zeros(2),
        off_track=False,
    )
    safe, _ = reward.compute(StepSignals(**base, collision=False))
    crashed, _ = reward.compute(StepSignals(**base, collision=True))
    assert safe > 0.0
    assert crashed < 0.0


def test_stuck_reset_waits_for_grace_and_patience_windows():
    termination = EpisodeTermination(
        {
            "collision_distance_m": 0.65,
            "collision_patience_steps": 3,
            "stuck_speed_mps": 0.15,
            "stuck_after_steps": 500,
            "stuck_patience_steps": 500,
            "target_laps": 1,
            "max_episode_steps": 15000,
        },
        off_track_distance_m=5.0,
    )
    result = None
    for step in range(1, 999):
        result = termination.update(
            step=step,
            lap_delta=0,
            min_clearance_m=2.0,
            speed_mps=0.0,
            track_distance_m=0.0,
        )
        assert result[0] is False

    result = termination.update(
        step=999,
        lap_delta=0,
        min_clearance_m=2.0,
        speed_mps=0.0,
        track_distance_m=0.0,
    )
    assert result[:3] == (True, False, "stuck")
