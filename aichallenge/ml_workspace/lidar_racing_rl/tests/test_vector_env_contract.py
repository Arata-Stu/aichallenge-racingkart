"""Contract tests for the LiDAR-only vector environment boundary."""

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from lidar_racing_rl.envs.vector_env import LidarRacingEnv, RacingEnvSettings


@struct.dataclass
class _FakeSimulatorState:
    rewards: jax.Array
    done: jax.Array
    step: jax.Array
    cartesian_states: jax.Array
    last_cartesian_states: jax.Array
    frenet_states: jax.Array
    last_frenet_states: jax.Array
    collisions: jax.Array
    num_laps: jax.Array
    scans: jax.Array
    prev_winding_vector: jax.Array
    accumulated_angles: jax.Array
    last_accumulated_angles: jax.Array


class _FakeTrack:
    has_boundaries = True
    left_widths = jnp.asarray([10.0, 10.0])
    right_widths = jnp.asarray([10.0, 10.0])

    @staticmethod
    def vmap_frenet_to_cartesian_jax(frenet: jax.Array) -> jax.Array:
        return frenet[:, jnp.array([0, 1, 2])]

    @staticmethod
    def is_off_track_frenet_jax(
        frenet: jax.Array,
        clearance: jax.Array,
    ) -> jax.Array:
        return jnp.abs(frenet[..., 1]) + clearance > 10.0


class _FakeArrayResult(NamedTuple):
    observations: jax.Array
    state: _FakeSimulatorState
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    infos: dict[str, Any]


class _FakeSimulator:
    agents = ("agent_0", "agent_1", "agent_2", "agent_3")
    track_length = jnp.asarray(100.0)
    winding_point = jnp.zeros((2,))
    track = _FakeTrack()

    @staticmethod
    def reset_array(key: jax.Array) -> tuple[jax.Array, _FakeSimulatorState]:
        del key
        cartesian = jnp.zeros((4, 5), dtype=jnp.float32)
        frenet = jnp.zeros((4, 3), dtype=jnp.float32)
        state = _FakeSimulatorState(
            rewards=jnp.zeros((4,)),
            done=jnp.zeros((4,), dtype=bool),
            step=jnp.asarray(0),
            cartesian_states=cartesian,
            last_cartesian_states=cartesian,
            frenet_states=frenet,
            last_frenet_states=frenet,
            collisions=jnp.zeros((4,), dtype=bool),
            num_laps=jnp.zeros((4,), dtype=jnp.int32),
            scans=jnp.full((4, 360), 20.0),
            prev_winding_vector=jnp.zeros((4, 2)),
            accumulated_angles=jnp.zeros((4,)),
            last_accumulated_angles=jnp.zeros((4,)),
        )
        # This raw observation deliberately contains forbidden GT; the wrapper
        # must discard it and expose only State.scans.
        return state.scans, state

    @staticmethod
    def reset_from_frenet_poses(
        key: jax.Array,
        frenet: jax.Array,
    ) -> tuple[jax.Array, _FakeSimulatorState]:
        _, state = _FakeSimulator.reset_array(key)
        cartesian = state.cartesian_states.at[:, [0, 1, 4]].set(frenet)
        state = state.replace(
            cartesian_states=cartesian,
            last_cartesian_states=cartesian,
            frenet_states=frenet,
            last_frenet_states=frenet,
        )
        return state.scans, state

    @staticmethod
    def step_env_array(
        key: jax.Array,
        state: _FakeSimulatorState,
        actions: jax.Array,
    ) -> _FakeArrayResult:
        del key
        ego_collision = actions[0, 0] > 0.5
        collisions = state.collisions.at[0].set(ego_collision)
        progress = jnp.full((4,), 0.1, dtype=jnp.float32)
        frenet = state.frenet_states.at[:, 0].add(progress)
        frenet = frenet.at[0, 1].set(
            jnp.where(actions[0, 1] > 2.0, 20.0, frenet[0, 1])
        )
        next_state = state.replace(
            step=state.step + 1,
            last_frenet_states=state.frenet_states,
            frenet_states=frenet,
            collisions=collisions,
            scans=jnp.full_like(state.scans, 20.0),
        )
        terminated = collisions
        truncated = jnp.zeros_like(collisions)
        return _FakeArrayResult(
            observations=next_state.scans,
            state=next_state,
            rewards=jnp.zeros((4,)),
            terminated=terminated,
            truncated=truncated,
            infos={},
        )


class _SingleAgentFakeSimulator:
    agents = ("agent_0",)
    track_length = jnp.asarray(100.0)
    track = _FakeTrack()

    @staticmethod
    def reset_array(key: jax.Array) -> tuple[jax.Array, _FakeSimulatorState]:
        del key
        raise AssertionError("the wrapper must use the external Frenet reset API")

    @staticmethod
    def reset_from_frenet_poses(
        key: jax.Array,
        frenet: jax.Array,
    ) -> tuple[jax.Array, _FakeSimulatorState]:
        del key
        cartesian = jnp.zeros((1, 5), dtype=jnp.float32)
        cartesian = cartesian.at[:, [0, 1, 4]].set(frenet)
        state = _FakeSimulatorState(
            rewards=jnp.zeros((1,)),
            done=jnp.zeros((1,), dtype=bool),
            step=jnp.asarray(0),
            cartesian_states=cartesian,
            last_cartesian_states=cartesian,
            frenet_states=frenet,
            last_frenet_states=frenet,
            collisions=jnp.zeros((1,), dtype=bool),
            num_laps=jnp.zeros((1,), dtype=jnp.int32),
            scans=jnp.full((1, 360), 20.0),
            prev_winding_vector=jnp.zeros((1, 2)),
            accumulated_angles=jnp.zeros((1,)),
            last_accumulated_angles=jnp.zeros((1,)),
        )
        return state.scans, state

    @staticmethod
    def step_env_array(
        key: jax.Array,
        state: _FakeSimulatorState,
        actions: jax.Array,
    ) -> _FakeArrayResult:
        del key, actions
        next_state = state.replace(
            step=state.step + 1,
            last_cartesian_states=state.cartesian_states,
            last_frenet_states=state.frenet_states,
        )
        return _FakeArrayResult(
            observations=next_state.scans,
            state=next_state,
            rewards=jnp.zeros((1,)),
            terminated=jnp.zeros((1,), dtype=bool),
            truncated=jnp.zeros((1,), dtype=bool),
            infos={},
        )


def _valid_settings() -> RacingEnvSettings:
    return RacingEnvSettings(
        num_agents=4,
        num_beams=360,
        frame_stack=4,
        field_of_view=1.5 * math.pi,
        range_min=0.001,
        range_max=30.0,
        vehicle_length=2.0,
        vehicle_width=1.45,
        max_steering_angle=0.64,
        min_acceleration=-3.2,
        max_acceleration=3.2,
        max_steps=1800,
        max_num_laps=1,
        reset_longitudinal_spacing=4.0,
        reset_lateral_jitter=0.25,
        reset_heading_jitter=0.02,
        progress_weight=1.0,
        collision_weight=10.0,
        off_track_weight=10.0,
        smoothness_weight=0.05,
        reverse_weight=1.0,
        relative_progress_enabled=True,
        pass_event_enabled=True,
        unsafe_contact_enabled=True,
        stalled_behind_vehicle_enabled=True,
        relative_progress_weight=0.5,
        pass_weight=5.0,
        unsafe_contact_weight=2.0,
        stalled_behind_weight=0.1,
        pass_behind_distance=0.5,
        pass_ahead_distance=1.0,
        pass_hold_steps=5,
        pass_cooldown_steps=100,
        unsafe_contact_distance=2.25,
        stalled_max_forward_gap=8.0,
        stalled_speed_threshold=0.25,
    )


def test_settings_accept_fixed_lidar_only_contract() -> None:
    _valid_settings().validate()


def test_enabled_step2_reward_requires_positive_weight() -> None:
    settings = _valid_settings()
    invalid = RacingEnvSettings(
        **{**settings.__dict__, "pass_weight": 0.0},
    )

    with pytest.raises(ValueError, match="enabled pass_event"):
        invalid.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_agents", 3, "either one or four"),
        ("num_beams", 1080, "exactly 360"),
        ("frame_stack", 3, "frame_stack=4"),
        ("field_of_view", math.pi, "270-degree"),
        ("range_max", 0.001, "LiDAR bounds"),
        ("range_max", 20.0, "range_max=30"),
        ("reset_longitudinal_spacing", 1.0, "at least one vehicle length"),
    ],
)
def test_settings_reject_boundary_violations(
    field: str, value: float, message: str
) -> None:
    settings = _valid_settings()
    invalid = RacingEnvSettings(
        **{**settings.__dict__, field: value},
    )
    with pytest.raises(ValueError, match=message):
        invalid.validate()


def test_actor_observation_contract_has_no_gt_fields() -> None:
    settings = _valid_settings()
    actor_observation = jnp.zeros(
        (settings.frame_stack, 2, settings.num_beams), dtype=jnp.float32
    )

    assert actor_observation.shape == (4, 2, 360)
    assert not isinstance(actor_observation, dict)


def test_single_agent_reset_uses_configured_external_frenet_jitter() -> None:
    base = _valid_settings()
    settings = RacingEnvSettings(
        **{
            **base.__dict__,
            "num_agents": 1,
            "relative_progress_enabled": False,
            "pass_event_enabled": False,
            "unsafe_contact_enabled": False,
            "stalled_behind_vehicle_enabled": False,
            "relative_progress_weight": 0.0,
            "pass_weight": 0.0,
            "unsafe_contact_weight": 0.0,
            "stalled_behind_weight": 0.0,
        },
    )
    environment = LidarRacingEnv(_SingleAgentFakeSimulator(), settings)

    state, observation = environment.reset(jax.random.key(0))

    frenet = state.simulator_state.frenet_states
    assert frenet.shape == (1, 3)
    assert observation.shape == (4, 2, 360)
    assert state.overtaking_state.armed_from_behind.shape == (0,)
    assert bool(jnp.all(jnp.abs(frenet[:, 1]) <= settings.reset_lateral_jitter))
    assert bool(jnp.all(jnp.abs(frenet[:, 2]) <= settings.reset_heading_jitter))

    result = environment.step(
        jax.random.key(1),
        state,
        jnp.zeros((2,)),
        jnp.empty((0, 2)),
    )
    assert float(result.diagnostics.nearest_opponent_distance) == 0.0
    assert int(result.diagnostics.ego_rank) == 1
    assert not bool(result.diagnostics.opponent_present)
    assert not bool(result.diagnostics.following_vehicle)
    assert not bool(result.diagnostics.collision_with_opponent)
    assert not bool(result.diagnostics.collision_with_wall)


def test_vector_step_resets_only_environment_whose_ego_terminated() -> None:
    environment = LidarRacingEnv(_FakeSimulator(), _valid_settings())
    reset_keys = jax.random.split(jax.random.key(1), 2)
    states, observations = environment.reset_batch(reset_keys)

    ego_actions = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    npc_actions = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    step_keys = jax.random.split(jax.random.key(2), 2)
    result = jax.jit(environment.step_batch)(
        step_keys,
        states,
        ego_actions,
        npc_actions,
    )

    assert observations.shape == (2, 4, 2, 360)
    assert result.observation.shape == (2, 4, 2, 360)
    assert result.transition_next_observation.shape == (2, 4, 2, 360)
    assert bool(result.terminated[0])
    assert not bool(result.terminated[1])
    assert bool(result.diagnostics.collision[0])
    assert not bool(result.diagnostics.collision_with_opponent[0])
    assert bool(result.diagnostics.collision_with_wall[0])
    assert not bool(result.diagnostics.npc_collision_without_ego[0])
    assert result.diagnostics.npc_collision_flags.shape == (2, 3)
    assert not bool(result.diagnostics.off_track[0])
    assert result.diagnostics.relative_progress.shape == (2,)
    assert result.diagnostics.pass_count.shape == (2,)
    assert result.diagnostics.minimum_npc_speed.shape == (2,)
    assert not bool(result.diagnostics.unsafe_contact[1])
    assert bool(result.diagnostics.stalled_behind_vehicle[1])
    assert bool(result.diagnostics.opponent_present[1])
    assert bool(result.diagnostics.following_vehicle[1])
    assert int(result.diagnostics.ego_rank[1]) == 4
    assert float(result.diagnostics.nearest_opponent_distance[1]) > 2.25
    assert int(result.state.simulator_state.step[0]) == 0
    assert int(result.state.simulator_state.step[1]) == 1


def test_collision_diagnostics_classify_current_opponent_obb_overlap() -> None:
    environment = LidarRacingEnv(_FakeSimulator(), _valid_settings())
    state, _ = environment.reset(jax.random.key(5))
    simulator_state = state.simulator_state
    overlapping_cartesian = simulator_state.cartesian_states.at[1, 0:2].set(
        simulator_state.cartesian_states[0, 0:2]
    )
    state = state.replace(
        simulator_state=simulator_state.replace(
            cartesian_states=overlapping_cartesian,
            last_cartesian_states=overlapping_cartesian,
        )
    )

    result = environment.step(
        jax.random.key(6),
        state,
        jnp.asarray([1.0, 0.0]),
        jnp.zeros((3, 2)),
    )

    assert bool(result.diagnostics.collision)
    assert bool(result.diagnostics.collision_with_opponent)
    assert not bool(result.diagnostics.collision_with_wall)


def test_off_track_is_a_true_termination_and_uses_terminal_diagnostics() -> None:
    environment = LidarRacingEnv(_FakeSimulator(), _valid_settings())
    state, _ = environment.reset(jax.random.key(3))
    result = environment.step(
        jax.random.key(4),
        state,
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((3, 2)),
    )

    assert bool(result.terminated)
    assert not bool(result.truncated)
    assert bool(result.diagnostics.off_track)
    assert not bool(result.diagnostics.collision)
    assert int(result.state.simulator_state.step) == 0
