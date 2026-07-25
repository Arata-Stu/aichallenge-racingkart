#include "rsu_laserscan_generator/rsu_laserscan_generator_node.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace rsu_laserscan_generator
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

std::vector<std::string> split_csv_line(const std::string & line)
{
  std::stringstream ss(line);
  std::string cell;
  std::vector<std::string> row;
  while (std::getline(ss, cell, ',')) {
    row.push_back(cell);
  }
  return row;
}

std::vector<int64_t> get_int_array_param(
  rclcpp::Node & node, const std::string & name, const std::vector<int64_t> & default_value = {})
{
  node.declare_parameter<std::vector<int64_t>>(name, default_value);
  return node.get_parameter(name).as_integer_array();
}

std::set<int64_t> to_set(const std::vector<int64_t> & values)
{
  return std::set<int64_t>(values.begin(), values.end());
}

std::string numbered_id(const std::string & prefix, const int index)
{
  std::ostringstream stream;
  stream << prefix << std::setw(2) << std::setfill('0') << index;
  return stream.str();
}
}  // namespace

RsuLaserScanGeneratorNode::RsuLaserScanGeneratorNode()
: Node("rsu_laserscan_generator_node")
{
  declare_and_get_params();
  load_walls_from_csv();
  load_rsus_from_params();

  if (rsus_.empty()) {
    RCLCPP_ERROR(get_logger(), "No RSU configs were loaded. Set rsu_ids in the parameter file.");
    rclcpp::shutdown();
    return;
  }

  for (auto & rsu : rsus_) {
    rsu.publisher = create_publisher<sensor_msgs::msg::LaserScan>(rsu.topic, 10);
  }

  if (debug_) {
    marker_publisher_ = create_publisher<MarkerArray>("~/debug/markers", rclcpp::QoS(1).transient_local());
    publish_debug_markers();
  }

  if (default_timer_hz_ <= 0.0) {
    RCLCPP_ERROR(get_logger(), "default_timer_hz must be positive.");
    rclcpp::shutdown();
    return;
  }

  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / default_timer_hz_),
    std::bind(&RsuLaserScanGeneratorNode::timer_callback, this));
}

void RsuLaserScanGeneratorNode::declare_and_get_params()
{
  declare_parameter<std::string>("csv_path", "/path/to/lane.csv");
  declare_parameter<std::string>("map_frame_id", "map");
  declare_parameter<double>("default_fov_deg", 90.0);
  declare_parameter<double>("default_max_range", 35.0);
  declare_parameter<double>("default_range_min", 0.05);
  declare_parameter<double>("default_timer_hz", 20.0);
  declare_parameter<int>("default_num_rays", 361);
  declare_parameter<int>("default_hit_rank", 2);
  declare_parameter<bool>("debug", true);
  declare_parameter<int>("rsu_count", 0);
  declare_parameter<std::string>("rsu_id_prefix", "curve_");

  get_parameter("csv_path", csv_path_);
  get_parameter("map_frame_id", map_frame_id_);
  get_parameter("default_fov_deg", default_fov_deg_);
  get_parameter("default_max_range", default_max_range_);
  get_parameter("default_range_min", default_range_min_);
  get_parameter("default_timer_hz", default_timer_hz_);
  get_parameter("default_num_rays", default_num_rays_);
  get_parameter("default_hit_rank", default_hit_rank_);
  get_parameter("debug", debug_);
}

void RsuLaserScanGeneratorNode::load_rsus_from_params()
{
  declare_parameter<std::vector<std::string>>("rsu_ids", std::vector<std::string>{});
  auto rsu_ids = get_parameter("rsu_ids").as_string_array();

  if (rsu_ids.empty()) {
    const int rsu_count = std::max(0, get_parameter("rsu_count").as_int());
    const auto rsu_id_prefix = get_parameter("rsu_id_prefix").as_string();
    for (int i = 1; i <= rsu_count; ++i) {
      rsu_ids.push_back(numbered_id(rsu_id_prefix, i));
    }
  }

  for (const auto & id : rsu_ids) {
    const std::string prefix = "rsus." + id + ".";
    RsuConfig rsu;
    rsu.id = id;

    declare_parameter<std::string>(prefix + "frame_id", "rsu_" + id + "_laser");
    declare_parameter<std::string>(prefix + "topic", "/rsu/" + id + "/scan");
    declare_parameter<double>(prefix + "x", 0.0);
    declare_parameter<double>(prefix + "y", 0.0);
    declare_parameter<double>(prefix + "z", 0.0);
    declare_parameter<double>(prefix + "yaw_deg", 0.0);
    declare_parameter<double>(prefix + "fov_deg", default_fov_deg_);
    declare_parameter<double>(prefix + "max_range", default_max_range_);
    declare_parameter<double>(prefix + "range_min", default_range_min_);
    declare_parameter<double>(prefix + "timer_hz", default_timer_hz_);
    declare_parameter<int>(prefix + "num_rays", default_num_rays_);
    declare_parameter<int>(prefix + "hit_rank", default_hit_rank_);
    declare_parameter<std::string>(prefix + "target_boundary", "any");

    get_parameter(prefix + "frame_id", rsu.frame_id);
    get_parameter(prefix + "topic", rsu.topic);
    get_parameter(prefix + "x", rsu.position.x);
    get_parameter(prefix + "y", rsu.position.y);
    get_parameter(prefix + "z", rsu.z);
    rsu.position = to_internal(rsu.position);

    double yaw_deg = 0.0;
    double fov_deg = default_fov_deg_;
    get_parameter(prefix + "yaw_deg", yaw_deg);
    get_parameter(prefix + "fov_deg", fov_deg);
    rsu.yaw_rad = yaw_deg * kPi / 180.0;
    rsu.fov_rad = fov_deg * kPi / 180.0;
    get_parameter(prefix + "max_range", rsu.max_range);
    get_parameter(prefix + "range_min", rsu.range_min);
    get_parameter(prefix + "timer_hz", rsu.timer_hz);
    get_parameter(prefix + "num_rays", rsu.num_rays);
    get_parameter(prefix + "hit_rank", rsu.hit_rank);
    get_parameter(prefix + "target_boundary", rsu.target_boundary);
    rsu.target_lanelet_ids = to_set(get_int_array_param(*this, prefix + "target_lanelet_ids"));
    rsu.target_way_ids = to_set(get_int_array_param(*this, prefix + "target_way_ids"));

    rsu.hit_rank = std::max(1, rsu.hit_rank);
    rsu.num_rays = std::max(2, rsu.num_rays);
    rsu.range_min = std::max(0.0, rsu.range_min);
    rsu.max_range = std::max(rsu.range_min, rsu.max_range);
    rsus_.push_back(rsu);

    RCLCPP_INFO(
      get_logger(), "Loaded RSU '%s': topic=%s hit_rank=%d target_boundary=%s",
      id.c_str(), rsu.topic.c_str(), rsu.hit_rank, rsu.target_boundary.c_str());
  }
}

void RsuLaserScanGeneratorNode::load_walls_from_csv()
{
  std::ifstream file(csv_path_);
  if (!file.is_open()) {
    RCLCPP_ERROR(get_logger(), "Failed to open CSV file: %s", csv_path_.c_str());
    rclcpp::shutdown();
    return;
  }

  std::map<int64_t, std::vector<std::pair<int64_t, WallSegment>>> way_points;
  std::string line;
  std::getline(file, line);

  while (std::getline(file, line)) {
    const auto row = split_csv_line(line);
    if (row.size() <= 6) {
      continue;
    }

    try {
      WallSegment point;
      point.lanelet_id = std::stoll(row[0]);
      point.way_id = std::stoll(row[1]);
      point.boundary_type = row[2];
      const int64_t seq_order = std::stoll(row[4]);
      point.start = {std::stod(row[5]), std::stod(row[6])};

      if (!is_offset_initialized_) {
        map_offset_ = point.start;
        is_offset_initialized_ = true;
        RCLCPP_INFO(
          get_logger(), "Map offset initialized to (%.3f, %.3f)", map_offset_.x, map_offset_.y);
      }
      point.start = to_internal(point.start);
      way_points[point.way_id].push_back({seq_order, point});
    } catch (const std::exception & error) {
      RCLCPP_WARN(get_logger(), "Could not parse CSV line: %s", error.what());
    }
  }

  for (const auto & [way_id, points] : way_points) {
    auto sorted = points;
    std::sort(sorted.begin(), sorted.end(), [](const auto & lhs, const auto & rhs) {
      return lhs.first < rhs.first;
    });
    for (size_t i = 0; i + 1 < sorted.size(); ++i) {
      WallSegment segment = sorted[i].second;
      segment.end = sorted[i + 1].second.start;
      segment.way_id = way_id;
      walls_.push_back(segment);
    }
  }

  RCLCPP_INFO(get_logger(), "Loaded %zu lane boundary segments.", walls_.size());
}

void RsuLaserScanGeneratorNode::timer_callback()
{
  for (const auto & rsu : rsus_) {
    publish_scan(rsu);
  }
}

void RsuLaserScanGeneratorNode::publish_scan(const RsuConfig & rsu)
{
  auto scan = std::make_unique<sensor_msgs::msg::LaserScan>();
  scan->header.stamp = now();
  scan->header.frame_id = rsu.frame_id;
  scan->angle_min = -rsu.fov_rad / 2.0;
  scan->angle_max = rsu.fov_rad / 2.0;
  scan->angle_increment = rsu.fov_rad / static_cast<double>(rsu.num_rays - 1);
  scan->time_increment = 0.0;
  scan->scan_time = rsu.timer_hz > 0.0 ? 1.0 / rsu.timer_hz : 1.0 / default_timer_hz_;
  scan->range_min = rsu.range_min;
  scan->range_max = rsu.max_range;
  scan->ranges.assign(rsu.num_rays, std::numeric_limits<float>::infinity());

  for (int i = 0; i < rsu.num_rays; ++i) {
    const double relative_angle = scan->angle_min + i * scan->angle_increment;
    const double ray_angle = rsu.yaw_rad + relative_angle;
    const Point2D ray_end{
      rsu.position.x + rsu.max_range * std::cos(ray_angle),
      rsu.position.y + rsu.max_range * std::sin(ray_angle)};

    std::vector<double> distances;
    for (const auto & wall : walls_) {
      if (!wall_matches_rsu(wall, rsu)) {
        continue;
      }
      const auto intersection =
        get_line_segment_intersection(rsu.position, ray_end, wall.start, wall.end);
      if (!intersection) {
        continue;
      }
      const double dx = intersection->x - rsu.position.x;
      const double dy = intersection->y - rsu.position.y;
      const double distance = std::sqrt(dx * dx + dy * dy);
      if (distance >= rsu.range_min && distance <= rsu.max_range) {
        distances.push_back(distance);
      }
    }

    if (static_cast<int>(distances.size()) >= rsu.hit_rank) {
      std::sort(distances.begin(), distances.end());
      scan->ranges[i] = static_cast<float>(distances[static_cast<size_t>(rsu.hit_rank - 1)]);
    }
  }

  rsu.publisher->publish(std::move(scan));
}

bool RsuLaserScanGeneratorNode::wall_matches_rsu(const WallSegment & wall, const RsuConfig & rsu) const
{
  if (rsu.target_boundary != "any" && wall.boundary_type != rsu.target_boundary) {
    return false;
  }
  if (!rsu.target_lanelet_ids.empty() && rsu.target_lanelet_ids.count(wall.lanelet_id) == 0) {
    return false;
  }
  if (!rsu.target_way_ids.empty() && rsu.target_way_ids.count(wall.way_id) == 0) {
    return false;
  }
  return true;
}

std::optional<Point2D> RsuLaserScanGeneratorNode::get_line_segment_intersection(
  Point2D p1, Point2D p2, Point2D p3, Point2D p4) const
{
  const double den = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
  if (std::abs(den) < 1e-9) {
    return std::nullopt;
  }

  const double t_num = (p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x);
  const double u_num = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x));
  const double t = t_num / den;
  const double u = u_num / den;
  if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
    return Point2D{p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)};
  }
  return std::nullopt;
}

Point2D RsuLaserScanGeneratorNode::to_internal(Point2D point) const
{
  return {point.x - map_offset_.x, point.y - map_offset_.y};
}

Point2D RsuLaserScanGeneratorNode::to_map(Point2D point) const
{
  return {point.x + map_offset_.x, point.y + map_offset_.y};
}

void RsuLaserScanGeneratorNode::publish_debug_markers()
{
  if (!marker_publisher_) {
    return;
  }

  MarkerArray array;

  Marker walls;
  walls.header.frame_id = map_frame_id_;
  walls.header.stamp = now();
  walls.ns = "rsu_target_walls";
  walls.id = 0;
  walls.type = Marker::LINE_LIST;
  walls.action = Marker::ADD;
  walls.pose.orientation.w = 1.0;
  walls.scale.x = 0.05;
  walls.color.g = 0.8f;
  walls.color.b = 1.0f;
  walls.color.a = 0.7f;
  for (const auto & wall : walls_) {
    const auto p1_map = to_map(wall.start);
    const auto p2_map = to_map(wall.end);
    geometry_msgs::msg::Point p1;
    geometry_msgs::msg::Point p2;
    p1.x = p1_map.x;
    p1.y = p1_map.y;
    p2.x = p2_map.x;
    p2.y = p2_map.y;
    walls.points.push_back(p1);
    walls.points.push_back(p2);
  }
  array.markers.push_back(walls);

  int marker_id = 1;
  for (const auto & rsu : rsus_) {
    const auto pos_map = to_map(rsu.position);

    Marker origin;
    origin.header.frame_id = map_frame_id_;
    origin.header.stamp = now();
    origin.ns = "rsu_origins";
    origin.id = marker_id++;
    origin.type = Marker::SPHERE;
    origin.action = Marker::ADD;
    origin.pose.position.x = pos_map.x;
    origin.pose.position.y = pos_map.y;
    origin.pose.position.z = rsu.z;
    origin.pose.orientation.w = 1.0;
    origin.scale.x = 0.4;
    origin.scale.y = 0.4;
    origin.scale.z = 0.4;
    origin.color.r = 1.0f;
    origin.color.g = 0.7f;
    origin.color.a = 0.9f;
    array.markers.push_back(origin);

    Marker fov;
    fov.header.frame_id = map_frame_id_;
    fov.header.stamp = now();
    fov.ns = "rsu_fov";
    fov.id = marker_id++;
    fov.type = Marker::LINE_LIST;
    fov.action = Marker::ADD;
    fov.pose.orientation.w = 1.0;
    fov.scale.x = 0.08;
    fov.color.r = 1.0f;
    fov.color.g = 0.4f;
    fov.color.a = 0.9f;

    for (const double edge : {-0.5, 0.5}) {
      const double angle = rsu.yaw_rad + edge * rsu.fov_rad;
      geometry_msgs::msg::Point start;
      geometry_msgs::msg::Point end;
      start.x = pos_map.x;
      start.y = pos_map.y;
      start.z = rsu.z;
      end.x = pos_map.x + rsu.max_range * std::cos(angle);
      end.y = pos_map.y + rsu.max_range * std::sin(angle);
      end.z = rsu.z;
      fov.points.push_back(start);
      fov.points.push_back(end);
    }
    array.markers.push_back(fov);
  }

  marker_publisher_->publish(array);
}

}  // namespace rsu_laserscan_generator

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rsu_laserscan_generator::RsuLaserScanGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
