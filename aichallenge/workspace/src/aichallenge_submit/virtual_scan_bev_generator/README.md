# virtual_scan_bev_generator

Efficiently converts synchronized ego VirtualScan topics into an ego-centric
multi-channel bird's-eye-view image. It intentionally keeps all cells beyond
the first static wall masked so a nearby section across a hairpin wall is not
treated as immediately reachable road.

## Inputs

- `/sensing/lidar/scan`: walls plus V2X-derived dynamic obstacles
- `/sensing/lidar/scan_without_obstacles`: static walls only

Both scans must have matching timestamps and LaserScan geometry.

## Outputs

- `/perception/virtual_scan_bev/image` (`sensor_msgs/Image`, `8UC8`)
- `/perception/virtual_scan_bev/debug_image` (`rgb8`, published lazily)
- `/perception/virtual_scan_bev/occupancy_grid` (`nav_msgs/OccupancyGrid`, published lazily)

The packed image is `height x width x 8`, interleaved by pixel:

| Channel | Meaning |
|---:|---|
| 0 | Statically reachable before the first wall |
| 1 | Observed free before the current obstacle hit |
| 2 | Static wall endpoint |
| 3 | Dynamic obstacle inferred from the scan difference |
| 4 | Region hidden behind a dynamic obstacle but before the wall |
| 5 | Region beyond the first static wall; never reachable in the local BEV |
| 6 | Cell covered by a LaserScan ray |
| 7 | Ego footprint |

Values are `0` or `255`. Grid axes follow the scan frame: `+x` forward and
`+y` left. The default grid covers 8 m behind and 20 m ahead, with 16 m on
each side, at 0.2 m/cell (`140 x 160`).

## Run

```bash
ros2 launch virtual_scan_bev_generator virtual_scan_bev_generator.launch.xml
```

For immediate inspection, add the occupancy grid topic to RViz or open the
debug image with `rqt_image_view`. BEV generation follows the 50 Hz VirtualScan
source by default. All publishers use a one-frame best-effort queue so an old
grid cannot build up behind the current LaserScan. Ray-to-cell lookup tables are rebuilt only if LaserScan geometry
changes; debug products are skipped when they have no subscribers.
