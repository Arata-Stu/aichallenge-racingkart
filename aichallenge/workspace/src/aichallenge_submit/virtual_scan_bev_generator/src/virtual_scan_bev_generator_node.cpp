#include "virtual_scan_bev_generator/bev_rasterizer.hpp"

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace virtual_scan_bev_generator
{

class VirtualScanBevGeneratorNode : public rclcpp::Node
{
public:
  VirtualScanBevGeneratorNode()
  : Node("virtual_scan_bev_generator_node"), rasterizer_(read_grid_config())
  {
    sync_tolerance_ns_ = static_cast<std::int64_t>(
      std::max(0.0, declare_parameter<double>("sync_tolerance_sec", 0.002)) * 1.0e9);
    const double max_publish_hz = declare_parameter<double>("max_publish_hz", 50.0);
    min_publish_period_ns_ = max_publish_hz > 0.0 ?
      static_cast<std::int64_t>(1.0e9 / max_publish_hz) : 0;

    // Keep only the newest local grid. A queued grid is already spatially stale
    // when RViz transforms it into the map frame at racing speed.
    const auto output_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    bev_publisher_ = create_publisher<sensor_msgs::msg::Image>("bev/image", output_qos);
    debug_publisher_ = create_publisher<sensor_msgs::msg::Image>("bev/debug_image", output_qos);
    grid_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("bev/occupancy_grid",
        output_qos);

    auto input_qos = rclcpp::SensorDataQoS();
    input_qos.keep_last(1);
    obstacle_subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", input_qos,
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(pair_mutex_);
          obstacle_scan_ = std::move(message);
        }
        try_process_pair();
      });
    static_subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan_without_obstacles", input_qos,
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr message) {
        {
          std::lock_guard<std::mutex> lock(pair_mutex_);
          static_scan_ = std::move(message);
        }
        try_process_pair();
      });

    const auto & config = rasterizer_.grid_config();
    RCLCPP_INFO(
      get_logger(),
      "VirtualScan BEV ready: x=[%.1f, %.1f] y=[%.1f, %.1f] resolution=%.2f, "
      "size=%zux%zu, output<=%.1f Hz",
      config.x_min, config.x_max, config.y_min, config.y_max, config.resolution,
      rasterizer_.width(), rasterizer_.height(), max_publish_hz);
  }

private:
  GridConfig read_grid_config()
  {
    GridConfig config;
    config.x_min = static_cast<float>(declare_parameter<double>("grid.x_min", config.x_min));
    config.x_max = static_cast<float>(declare_parameter<double>("grid.x_max", config.x_max));
    config.y_min = static_cast<float>(declare_parameter<double>("grid.y_min", config.y_min));
    config.y_max = static_cast<float>(declare_parameter<double>("grid.y_max", config.y_max));
    config.resolution = static_cast<float>(
      declare_parameter<double>("grid.resolution", config.resolution));
    config.dynamic_threshold = static_cast<float>(
      declare_parameter<double>("dynamic_threshold", config.dynamic_threshold));
    config.endpoint_radius = static_cast<float>(
      declare_parameter<double>("endpoint_radius", config.endpoint_radius));
    config.ego_radius = static_cast<float>(declare_parameter<double>("ego_radius",
        config.ego_radius));
    return config;
  }

  static std::int64_t stamp_ns(const sensor_msgs::msg::LaserScan & scan)
  {
    return rclcpp::Time(scan.header.stamp).nanoseconds();
  }

  void try_process_pair()
  {
    sensor_msgs::msg::LaserScan::ConstSharedPtr obstacle;
    sensor_msgs::msg::LaserScan::ConstSharedPtr static_only;
    {
      std::lock_guard<std::mutex> lock(pair_mutex_);
      if (!obstacle_scan_ || !static_scan_) {
        return;
      }
      const auto obstacle_stamp = stamp_ns(*obstacle_scan_);
      const auto static_stamp = stamp_ns(*static_scan_);
      const auto delta = std::llabs(obstacle_stamp - static_stamp);
      if (delta > sync_tolerance_ns_) {
        if (obstacle_stamp < static_stamp) {
          obstacle_scan_.reset();
        } else {
          static_scan_.reset();
        }
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Waiting for synchronized VirtualScans (current delta %.3f ms)",
          static_cast<double>(delta) / 1.0e6);
        return;
      }
      obstacle = std::move(obstacle_scan_);
      static_only = std::move(static_scan_);
      obstacle_scan_.reset();
      static_scan_.reset();
    }

    std::lock_guard<std::mutex> process_lock(process_mutex_);
    const auto current_stamp = std::max(stamp_ns(*obstacle), stamp_ns(*static_only));
    if (last_publish_stamp_ns_ > 0 && current_stamp < last_publish_stamp_ns_) {
      // Simulation time can jump backwards when a scenario restarts.
      last_publish_stamp_ns_ = 0;
      next_publish_stamp_ns_ = 0;
    } else if (last_publish_stamp_ns_ > 0 && current_stamp == last_publish_stamp_ns_) {
      return;
    }
    if (min_publish_period_ns_ > 0 && next_publish_stamp_ns_ > 0 &&
      current_stamp < next_publish_stamp_ns_)
    {
      return;
    }

    if (!same_geometry(*obstacle, *static_only)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "scan and scan_without_obstacles have different LaserScan geometry");
      return;
    }

    const ScanGeometry geometry{
      obstacle->angle_min,
      obstacle->angle_increment,
      obstacle->range_min,
      obstacle->range_max,
      obstacle->ranges.size()};
    if (!geometry_configured_ || !same_geometry(geometry, rasterizer_.scan_geometry())) {
      try {
        rasterizer_.configure(geometry);
        geometry_configured_ = true;
        RCLCPP_INFO(
          get_logger(), "Precomputed BEV lookup for %zu rays (FOV %.1f deg)",
          geometry.ray_count,
          geometry.angle_increment * static_cast<float>(geometry.ray_count - 1U) *
          180.0F / static_cast<float>(M_PI));
      } catch (const std::exception & error) {
        RCLCPP_ERROR(get_logger(), "Cannot configure BEV rasterizer: %s", error.what());
        return;
      }
    }

    const auto started = std::chrono::steady_clock::now();
    try {
      rasterizer_.rasterize(obstacle->ranges, static_only->ranges);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000, "BEV rasterization failed: %s", error.what());
      return;
    }

    publish_bev(*obstacle);
    last_publish_stamp_ns_ = current_stamp;
    if (min_publish_period_ns_ > 0) {
      if (next_publish_stamp_ns_ <= 0) {
        next_publish_stamp_ns_ = current_stamp + min_publish_period_ns_;
      } else {
        while (next_publish_stamp_ns_ <= current_stamp) {
          next_publish_stamp_ns_ += min_publish_period_ns_;
        }
      }
    }
    ++published_frames_;
    accumulated_processing_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
    if (published_frames_ % 200U == 0U) {
      RCLCPP_INFO(
        get_logger(), "BEV processing average %.3f ms/frame (%zu frames)",
        accumulated_processing_ms_ / static_cast<double>(published_frames_), published_frames_);
    }
  }

  static bool same_geometry(
    const sensor_msgs::msg::LaserScan & lhs, const sensor_msgs::msg::LaserScan & rhs)
  {
    constexpr float tolerance = 1.0e-5F;
    return lhs.ranges.size() == rhs.ranges.size() &&
           std::abs(lhs.angle_min - rhs.angle_min) <= tolerance &&
           std::abs(lhs.angle_increment - rhs.angle_increment) <= tolerance &&
           std::abs(lhs.range_min - rhs.range_min) <= tolerance &&
           std::abs(lhs.range_max - rhs.range_max) <= tolerance;
  }

  static bool same_geometry(const ScanGeometry & lhs, const ScanGeometry & rhs)
  {
    constexpr float tolerance = 1.0e-5F;
    return lhs.ray_count == rhs.ray_count &&
           std::abs(lhs.angle_min - rhs.angle_min) <= tolerance &&
           std::abs(lhs.angle_increment - rhs.angle_increment) <= tolerance &&
           std::abs(lhs.range_min - rhs.range_min) <= tolerance &&
           std::abs(lhs.range_max - rhs.range_max) <= tolerance;
  }

  void publish_bev(const sensor_msgs::msg::LaserScan & source)
  {
    sensor_msgs::msg::Image packed;
    packed.header = source.header;
    packed.height = static_cast<std::uint32_t>(rasterizer_.height());
    packed.width = static_cast<std::uint32_t>(rasterizer_.width());
    packed.encoding = "8UC8";
    packed.is_bigendian = false;
    packed.step = packed.width * static_cast<std::uint32_t>(kBevChannelCount);
    packed.data = rasterizer_.data();
    bev_publisher_->publish(std::move(packed));

    if (debug_publisher_->get_subscription_count() > 0U) {
      publish_debug_image(source);
    }
    if (grid_publisher_->get_subscription_count() > 0U) {
      publish_occupancy_grid(source);
    }
  }

  void publish_debug_image(const sensor_msgs::msg::LaserScan & source)
  {
    sensor_msgs::msg::Image image;
    image.header = source.header;
    image.height = static_cast<std::uint32_t>(rasterizer_.height());
    image.width = static_cast<std::uint32_t>(rasterizer_.width());
    image.encoding = "rgb8";
    image.is_bigendian = false;
    image.step = image.width * 3U;
    image.data.assign(static_cast<std::size_t>(image.step) * image.height, 0U);

    const auto & bev = rasterizer_.data();
    const std::size_t cells = rasterizer_.width() * rasterizer_.height();
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const auto channel = [&bev, cell](const BevChannel value) {
          return bev[cell * kBevChannelCount + static_cast<std::size_t>(value)] != 0U;
        };
      std::array<std::uint8_t, 3> color{0U, 0U, 0U};
      if (channel(BevChannel::kRayCovered)) {
        color = {18U, 22U, 24U};
      }
      if (channel(BevChannel::kWallOccluded)) {
        color = {18U, 28U, 55U};
      }
      if (channel(BevChannel::kReachable)) {
        color = {20U, 48U, 38U};
      }
      if (channel(BevChannel::kObservedFree)) {
        color = {32U, 96U, 68U};
      }
      if (channel(BevChannel::kObstacleOccluded)) {
        color = {104U, 42U, 130U};
      }
      if (channel(BevChannel::kStaticWall)) {
        color = {235U, 235U, 225U};
      }
      if (channel(BevChannel::kDynamicObstacle)) {
        color = {255U, 70U, 45U};
      }
      if (channel(BevChannel::kEgo)) {
        color = {30U, 220U, 255U};
      }
      image.data[cell * 3U] = color[0];
      image.data[cell * 3U + 1U] = color[1];
      image.data[cell * 3U + 2U] = color[2];
    }
    debug_publisher_->publish(std::move(image));
  }

  void publish_occupancy_grid(const sensor_msgs::msg::LaserScan & source)
  {
    nav_msgs::msg::OccupancyGrid grid;
    grid.header = source.header;
    grid.info.map_load_time = source.header.stamp;
    grid.info.resolution = rasterizer_.grid_config().resolution;
    grid.info.width = static_cast<std::uint32_t>(rasterizer_.width());
    grid.info.height = static_cast<std::uint32_t>(rasterizer_.height());
    grid.info.origin.position.x = rasterizer_.grid_config().x_min;
    grid.info.origin.position.y = rasterizer_.grid_config().y_min;
    grid.info.origin.orientation.w = 1.0;
    grid.data.assign(rasterizer_.width() * rasterizer_.height(), -1);

    const auto & bev = rasterizer_.data();
    for (std::size_t cell = 0; cell < grid.data.size(); ++cell) {
      const auto channel = [&bev, cell](const BevChannel value) {
          return bev[cell * kBevChannelCount + static_cast<std::size_t>(value)] != 0U;
        };
      if (channel(BevChannel::kObservedFree)) {
        grid.data[cell] = 0;
      }
      if (channel(BevChannel::kDynamicObstacle)) {
        grid.data[cell] = 80;
      }
      if (channel(BevChannel::kStaticWall)) {
        grid.data[cell] = 100;
      }
    }
    grid_publisher_->publish(std::move(grid));
  }

  BevRasterizer rasterizer_;
  bool geometry_configured_{false};
  std::int64_t sync_tolerance_ns_{0};
  std::int64_t min_publish_period_ns_{0};
  std::int64_t last_publish_stamp_ns_{0};
  std::int64_t next_publish_stamp_ns_{0};
  std::size_t published_frames_{0U};
  double accumulated_processing_ms_{0.0};

  std::mutex pair_mutex_;
  std::mutex process_mutex_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr obstacle_scan_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr static_scan_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr obstacle_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr static_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_publisher_;
};

}  // namespace virtual_scan_bev_generator

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<virtual_scan_bev_generator::VirtualScanBevGeneratorNode>());
  rclcpp::shutdown();
  return 0;
}
