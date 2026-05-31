#include "laserscan_generator/occupancy_map_scan_generator_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/time.h>
#include <tf2/utils.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
}

OccupancyMapScanGeneratorNode::OccupancyMapScanGeneratorNode()
: Node("occupancy_map_scan_generator_node")
{
  declare_and_get_params();
  if (!validate_params() || !load_map()) {
    rclcpp::shutdown();
    return;
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  const double angle_min = angle_min_deg_ * kDegToRad;
  const double angle_max = angle_max_deg_ * kDegToRad;
  const double angle_increment = (angle_max - angle_min) / static_cast<double>(num_rays_ - 1);
  relative_angles_.resize(static_cast<size_t>(num_rays_));
  for (int i = 0; i < num_rays_; ++i) {
    relative_angles_[static_cast<size_t>(i)] = angle_min + angle_increment * static_cast<double>(i);
  }

  scan_publisher_ =
    this->create_publisher<sensor_msgs::msg::LaserScan>(output_topic_, rclcpp::SensorDataQoS());

  if (debug_) {
    debug_hit_points_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "~/debug/hit_points", rclcpp::QoS(1));
  }

  const auto timer_period = std::chrono::duration<double>(1.0 / timer_hz_);
  timer_ = this->create_wall_timer(timer_period, [this]() { timer_callback(); });

  RCLCPP_INFO(
    this->get_logger(),
    "Occupancy map scan generator started: map=%s, size=%dx%d, resolution=%.4f, scan_frame=%s",
    map_path_.c_str(), map_gray_.cols, map_gray_.rows, resolution_, scan_frame_id_.c_str());
}

void OccupancyMapScanGeneratorNode::declare_and_get_params()
{
  this->declare_parameter<std::string>("map_path", "");
  this->declare_parameter<std::string>("map_frame_id", "map");
  this->declare_parameter<std::string>("scan_frame_id", "base_link");
  this->declare_parameter<std::string>("output_topic", "virtual_scan");

  this->declare_parameter<double>("resolution", 0.05);
  this->declare_parameter<double>("origin_x", 0.0);
  this->declare_parameter<double>("origin_y", 0.0);
  this->declare_parameter<bool>("invert_y_axis", true);

  this->declare_parameter<double>("timer_hz", 20.0);
  this->declare_parameter<double>("angle_min_deg", -135.0);
  this->declare_parameter<double>("angle_max_deg", 135.0);
  this->declare_parameter<int>("num_rays", 1080);
  this->declare_parameter<double>("range_min", 0.05);
  this->declare_parameter<double>("range_max", 30.0);
  this->declare_parameter<double>("ray_step", 0.0);
  this->declare_parameter<int>("occupied_threshold", 20);
  this->declare_parameter<bool>("debug", false);

  this->get_parameter("map_path", map_path_);
  this->get_parameter("map_frame_id", map_frame_id_);
  this->get_parameter("scan_frame_id", scan_frame_id_);
  this->get_parameter("output_topic", output_topic_);
  this->get_parameter("resolution", resolution_);
  this->get_parameter("origin_x", origin_x_);
  this->get_parameter("origin_y", origin_y_);
  this->get_parameter("invert_y_axis", invert_y_axis_);
  this->get_parameter("timer_hz", timer_hz_);
  this->get_parameter("angle_min_deg", angle_min_deg_);
  this->get_parameter("angle_max_deg", angle_max_deg_);
  this->get_parameter("num_rays", num_rays_);
  this->get_parameter("range_min", range_min_);
  this->get_parameter("range_max", range_max_);
  this->get_parameter("ray_step", ray_step_);
  this->get_parameter("occupied_threshold", occupied_threshold_);
  this->get_parameter("debug", debug_);

  if (ray_step_ <= 0.0) {
    ray_step_ = resolution_ * 0.5;
  }
}

bool OccupancyMapScanGeneratorNode::validate_params() const
{
  if (map_path_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "map_path is empty.");
    return false;
  }
  if (resolution_ <= 0.0) {
    RCLCPP_FATAL(this->get_logger(), "resolution must be positive.");
    return false;
  }
  if (timer_hz_ <= 0.0) {
    RCLCPP_FATAL(this->get_logger(), "timer_hz must be positive.");
    return false;
  }
  if (num_rays_ < 2) {
    RCLCPP_FATAL(this->get_logger(), "num_rays must be at least 2.");
    return false;
  }
  if (angle_max_deg_ <= angle_min_deg_) {
    RCLCPP_FATAL(this->get_logger(), "angle_max_deg must be greater than angle_min_deg.");
    return false;
  }
  if (range_min_ < 0.0 || range_max_ <= range_min_) {
    RCLCPP_FATAL(this->get_logger(), "range_min/range_max are invalid.");
    return false;
  }
  if (ray_step_ <= 0.0) {
    RCLCPP_FATAL(this->get_logger(), "ray_step must be positive.");
    return false;
  }
  if (occupied_threshold_ < 0 || occupied_threshold_ > 255) {
    RCLCPP_FATAL(this->get_logger(), "occupied_threshold must be in [0, 255].");
    return false;
  }
  return true;
}

bool OccupancyMapScanGeneratorNode::load_map()
{
  map_gray_ = cv::imread(map_path_, cv::IMREAD_GRAYSCALE);
  if (map_gray_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "Failed to read map image: %s", map_path_.c_str());
    return false;
  }
  return true;
}

void OccupancyMapScanGeneratorNode::timer_callback()
{
  geometry_msgs::msg::TransformStamped map_to_scan_frame;
  try {
    map_to_scan_frame = tf_buffer_->lookupTransform(
      map_frame_id_, scan_frame_id_, tf2::TimePointZero, tf2::durationFromSec(0.05));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000, "Waiting for TF %s -> %s: %s",
      map_frame_id_.c_str(), scan_frame_id_.c_str(), ex.what());
    return;
  }

  const auto stamp = this->get_clock()->now();
  const double x = map_to_scan_frame.transform.translation.x;
  const double y = map_to_scan_frame.transform.translation.y;
  const double yaw = tf2::getYaw(map_to_scan_frame.transform.rotation);

  auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
  scan_msg->header.stamp = stamp;
  scan_msg->header.frame_id = scan_frame_id_;
  scan_msg->angle_min = static_cast<float>(angle_min_deg_ * kDegToRad);
  scan_msg->angle_max = static_cast<float>(angle_max_deg_ * kDegToRad);
  scan_msg->angle_increment =
    static_cast<float>((scan_msg->angle_max - scan_msg->angle_min) / static_cast<double>(num_rays_ - 1));
  scan_msg->time_increment = 0.0;
  scan_msg->scan_time = static_cast<float>(1.0 / timer_hz_);
  scan_msg->range_min = static_cast<float>(range_min_);
  scan_msg->range_max = static_cast<float>(range_max_);
  scan_msg->ranges.assign(static_cast<size_t>(num_rays_), std::numeric_limits<float>::infinity());

  std::vector<geometry_msgs::msg::Point> hit_points;
  if (debug_) {
    hit_points.reserve(static_cast<size_t>(num_rays_));
  }

  for (int i = 0; i < num_rays_; ++i) {
    const float range = raycast(x, y, yaw, relative_angles_[static_cast<size_t>(i)]);
    scan_msg->ranges[static_cast<size_t>(i)] = range;

    if (debug_ && std::isfinite(range)) {
      const double ray_yaw = yaw + relative_angles_[static_cast<size_t>(i)];
      geometry_msgs::msg::Point point;
      point.x = x + static_cast<double>(range) * std::cos(ray_yaw);
      point.y = y + static_cast<double>(range) * std::sin(ray_yaw);
      point.z = map_to_scan_frame.transform.translation.z;
      hit_points.push_back(point);
    }
  }

  scan_publisher_->publish(std::move(scan_msg));

  if (debug_) {
    publish_debug_points(hit_points, stamp);
  }
}

float OccupancyMapScanGeneratorNode::raycast(
  const double origin_x, const double origin_y, const double yaw, const double relative_angle) const
{
  const double ray_yaw = yaw + relative_angle;
  const double cos_yaw = std::cos(ray_yaw);
  const double sin_yaw = std::sin(ray_yaw);

  bool has_been_inside_map = false;
  for (double range = range_min_; range <= range_max_; range += ray_step_) {
    const double world_x = origin_x + range * cos_yaw;
    const double world_y = origin_y + range * sin_yaw;

    PixelIndex pixel;
    if (!world_to_pixel(world_x, world_y, pixel)) {
      if (has_been_inside_map) {
        break;
      }
      continue;
    }

    has_been_inside_map = true;
    if (is_occupied(pixel)) {
      return static_cast<float>(range);
    }
  }

  return std::numeric_limits<float>::infinity();
}

bool OccupancyMapScanGeneratorNode::world_to_pixel(
  const double world_x, const double world_y, PixelIndex & pixel) const
{
  const int px = static_cast<int>(std::floor((world_x - origin_x_) / resolution_));
  const int map_y = static_cast<int>(std::floor((world_y - origin_y_) / resolution_));
  const int py = invert_y_axis_ ? (map_gray_.rows - 1 - map_y) : map_y;

  if (px < 0 || px >= map_gray_.cols || py < 0 || py >= map_gray_.rows) {
    return false;
  }

  pixel.x = px;
  pixel.y = py;
  return true;
}

bool OccupancyMapScanGeneratorNode::is_occupied(const PixelIndex & pixel) const
{
  const auto intensity = static_cast<int>(map_gray_.at<unsigned char>(pixel.y, pixel.x));
  return intensity <= occupied_threshold_;
}

void OccupancyMapScanGeneratorNode::publish_debug_points(
  const std::vector<geometry_msgs::msg::Point> & hit_points, const rclcpp::Time & stamp)
{
  if (!debug_hit_points_publisher_) {
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = map_frame_id_;
  marker.header.stamp = stamp;
  marker.ns = "occupancy_map_scan_hits";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::POINTS;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.08;
  marker.scale.y = 0.08;
  marker.color.r = 1.0;
  marker.color.g = 0.1;
  marker.color.b = 0.1;
  marker.color.a = 0.9;
  marker.points = hit_points;
  debug_hit_points_publisher_->publish(marker);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OccupancyMapScanGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
