"""Construction of the pinned F1TENTH Gym JAX environment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any


def _required(config: Mapping[str, Any], key: str) -> Any:
    try:
        return config[key]
    except KeyError as exc:
        raise KeyError(f"required environment configuration is missing: {key}") from exc


def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = _required(config, key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"required environment configuration '{key}' must be an integer")
    return int(value)


def make_f1tenth_env(
    env_config: Mapping[str, Any], vehicle_config: Mapping[str, Any]
) -> Any:
    """Create the four-vehicle scan environment from resolved YAML mappings.

    The upstream observation contains ground-truth dynamics even when
    ``observe_others`` is disabled.  Callers must therefore use ``state.scans``
    through the LiDAR-only wrapper and must not feed the raw observation dict to
    the Actor or Critic.
    """

    domain_randomization = env_config.get("domain_randomization")
    if isinstance(domain_randomization, Mapping) and domain_randomization.get(
        "enabled"
    ) is True:
        raise NotImplementedError(
            "AWSIM-calibrated vehicle-response domain randomization is not "
            "connected to F1TENTH Gym JAX yet"
        )

    import jax.numpy as jnp
    from f1tenth_gym_jax import make

    from lidar_racing_rl.geometry.dynamic_scan import dynamic_vehicle_scan

    simulator_config = _required(env_config, "simulator")
    lidar_config = _required(env_config, "lidar")
    episode_config = _required(env_config, "episode")
    vehicle = vehicle_config.get("vehicle", vehicle_config)

    num_agents = _required_int(env_config, "num_agents")
    if num_agents not in (1, 4):
        raise ValueError("the supported stages require either one or four vehicles")

    map_name = str(_required(simulator_config, "map_name"))
    timestep_ratio = _required_int(simulator_config, "timestep_ratio")
    max_steps = _required_int(episode_config, "max_steps")
    num_beams = _required_int(lidar_config, "num_beams")
    max_num_laps = _required_int(episode_config, "max_num_laps")
    field_of_view = float(_required(lidar_config, "field_of_view"))
    max_range = float(_required(lidar_config, "range_max"))
    if timestep_ratio < 1 or max_steps < 1 or max_num_laps < 1:
        raise ValueError("simulator ratio and episode limits must be positive")
    if num_beams != 360:
        raise ValueError("initial Actor/runtime contract requires num_beams=360")
    supported_sensor_profiles = (
        (1.5 * math.pi, 30.0),
        (math.pi, 25.0),
    )
    if not any(
        math.isclose(field_of_view, expected_fov, rel_tol=0.0, abs_tol=1.0e-9)
        and math.isclose(max_range, expected_range, rel_tol=0.0, abs_tol=1.0e-9)
        for expected_fov, expected_range in supported_sensor_profiles
    ):
        raise ValueError(
            "LiDAR profile must be legacy 270-degree/30m or AWSIM e2e 180-degree/25m"
        )
    env_id = (
        f"{map_name}_{num_agents}_scan_collision_progress_"
        f"acceleration+steeringangle_{timestep_ratio}_{max_steps}_v0"
    )

    acceleration_limit = max(
        abs(float(_required(vehicle, "min_acceleration"))),
        abs(float(_required(vehicle, "max_acceleration"))),
    )
    wheelbase = float(_required(vehicle, "wheelbase"))
    front_axle_distance = float(
        vehicle.get("front_axle_distance", 0.5 * wheelbase)
    )
    rear_axle_distance = float(
        vehicle.get("rear_axle_distance", wheelbase - front_axle_distance)
    )
    if not math.isclose(
        front_axle_distance + rear_axle_distance,
        wheelbase,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("front and rear axle distances must sum to wheelbase")
    minimum_steering = float(_required(vehicle, "min_steering_angle"))
    maximum_steering = float(_required(vehicle, "max_steering_angle"))
    if not math.isclose(
        minimum_steering,
        -maximum_steering,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("initial normalized action contract requires symmetric steering")

    beam_angles = jnp.linspace(
        -0.5 * field_of_view,
        0.5 * field_of_view,
        num_beams,
        dtype=jnp.float32,
    )
    vehicle_dimensions = jnp.asarray(
        [float(_required(vehicle, "length")), float(_required(vehicle, "width"))],
        dtype=jnp.float32,
    )

    def dynamic_vehicle_hook(key: Any, state: Any, current_ranges: Any) -> Any:
        """Render updated GT poses only into sensor-space dynamic ranges."""

        del key, current_ranges
        # GT is consumed solely to synthesize what each LiDAR would measure.
        poses = state.cartesian_states[:, jnp.asarray([0, 1, 4])]
        return dynamic_vehicle_scan(
            poses,
            vehicle_dimensions,
            beam_angles,
            max_range,
        )

    return make(
        env_id,
        observe_others=False,
        timestep=float(_required(simulator_config, "physics_timestep")),
        num_beams=num_beams,
        fov=field_of_view,
        max_range=max_range,
        max_num_laps=max_num_laps,
        length=float(_required(vehicle, "length")),
        width=float(_required(vehicle, "width")),
        lf=front_axle_distance,
        lr=rear_axle_distance,
        s_min=minimum_steering,
        s_max=maximum_steering,
        a_max=acceleration_limit,
        v_min=float(_required(vehicle, "min_velocity")),
        v_max=float(_required(vehicle, "max_velocity")),
        external_scan_hook=dynamic_vehicle_hook,
        scan_only_observation=True,
    )
