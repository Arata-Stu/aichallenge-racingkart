# rsu_laserscan_generator

Fixed roadside-unit virtual LiDAR generator.

This package is separate from `laserscan_generator`. The original generator
uses the vehicle pose as the LiDAR origin. This package reads fixed RSU poses
from ROS parameters and publishes one `sensor_msgs/msg/LaserScan` topic per RSU.

Key parameters:

- `rsu_ids`: list of RSU IDs to load from `rsus.<id>`.
- `rsu_count`: if `rsu_ids` is empty, auto-generates `curve_01 ... curve_N`.
- `rsu_id_prefix`: prefix used by `rsu_count`; default is `curve_`.
- `rsus.<id>.x/y/yaw_deg`: RSU pose in the map coordinate system used by `lane.csv`.
- `rsus.<id>.fov_deg`: horizontal scan field of view.
- `rsus.<id>.hit_rank`: `1` uses the nearest hit, `2` skips the nearest hit and uses the second hit.
- `rsus.<id>.target_boundary`: `any`, `left`, or `right`.
- `rsus.<id>.target_lanelet_ids` / `target_way_ids`: optional target filters.

For inside-corner RSUs, start with `hit_rank: 2` and a narrowed FOV. If the
second-hit heuristic is unstable at a specific corner, switch that RSU to
`target_boundary: left/right` or set `target_way_ids`.
