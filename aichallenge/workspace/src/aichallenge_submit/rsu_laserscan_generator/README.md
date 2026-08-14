# rsu_laserscan_generator

Fixed roadside-unit virtual LiDAR generator.

This package is separate from `laserscan_generator`. The original generator
uses the vehicle pose as the LiDAR origin. This package reads fixed RSU poses
from ROS parameters and publishes one `sensor_msgs/msg/LaserScan` topic per RSU.
It also subscribes to `v2x_msgs/msg/V2XVehiclePositionArray` and inserts current
vehicle positions into each scan as oriented rectangular dynamic obstacles.
By default it also publishes static `map -> <RSU frame>` transforms from the
configured RSU poses, so the scans can be displayed directly in RViz.

Key parameters:

- `rsu_ids`: list of RSU IDs to load from `rsus.<id>`.
- `rsu_count`: if `rsu_ids` is empty, auto-generates `curve_01 ... curve_N`.
- `rsu_id_prefix`: prefix used by `rsu_count`; default is `curve_`.
- `rsus.<id>.x/y/yaw_deg`: RSU pose in the map coordinate system used by `lane.csv`.
- `rsus.<id>.fov_deg`: horizontal scan field of view.
- `rsus.<id>.hit_rank`: `1` uses the nearest hit, `2` skips the nearest hit and uses the second hit.
- `rsus.<id>.target_boundary`: `any`, `left`, or `right`.
- `rsus.<id>.target_lanelet_ids` / `target_way_ids`: optional target filters.
- `enable_v2x_vehicles`: enables dynamic vehicle intersections; default is `true`.
- `v2x_topic`: vehicle-position array topic; default is `/v2x/vehicle_positions`.
- `v2x_vehicle_length` / `v2x_vehicle_width`: rectangular vehicle footprint in meters;
  defaults are the official simulator dimensions, `2.0` m by `1.45` m.
- `v2x_heading_min_displacement`: minimum position change used to update the vehicle
  heading. The previous heading is retained while stopped.
- `v2x_timeout_sec`: removes vehicles from scans after the V2X input becomes stale;
  `0.0` disables the timeout.
- `publish_static_tf`: publishes one static transform per RSU; default is `true`.

For inside-corner RSUs, start with `hit_rank: 2` and a narrowed FOV. If the
second-hit heuristic is unstable at a specific corner, switch that RSU to
`target_boundary: left/right` or set `target_way_ids`.

V2X vehicle coordinates must use `map_frame_id` (or have an empty frame ID).
For each ray, wall intersections still use the configured `hit_rank`; the
nearest V2X vehicle intersection is then combined with that wall result.

## RViz configuration

The RSU displays are defined in
`aichallenge_submit_launch/config/autoware-rsu.rviz`. The upstream
`aichallenge_system_launch/config/autoware.rviz` is intentionally left
unchanged to reduce merge conflicts. In simulation mode,
`aichallenge_system.launch.xml` selects the RSU-specific configuration.
