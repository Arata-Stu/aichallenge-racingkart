#ifndef RSU_LASERSCAN_GENERATOR_NODE_HPP_
#define RSU_LASERSCAN_GENERATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace rsu_laserscan_generator
{

struct Point2D
{
  double x{0.0};
  double y{0.0};
};

struct WallSegment
{
  int64_t lanelet_id{0};
  int64_t way_id{0};
  std::string boundary_type;
  Point2D start;
  Point2D end;
};

struct RsuConfig
{
  std::string id;
  std::string frame_id;
  std::string topic;
  Point2D position;
  double z{0.0};
  double yaw_rad{0.0};
  double fov_rad{0.0};
  double max_range{0.0};
  double range_min{0.0};
  double timer_hz{0.0};
  int num_rays{0};
  int hit_rank{1};
  std::string target_boundary{"any"};
  std::set<int64_t> target_lanelet_ids;
  std::set<int64_t> target_way_ids;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher;
};

class RsuLaserScanGeneratorNode : public rclcpp::Node
{
public:
  RsuLaserScanGeneratorNode();

private:
  using Marker = visualization_msgs::msg::Marker;
  using MarkerArray = visualization_msgs::msg::MarkerArray;

  std::string csv_path_;
  std::string map_frame_id_;
  double default_fov_deg_{90.0};
  double default_max_range_{35.0};
  double default_range_min_{0.05};
  double default_timer_hz_{20.0};
  int default_num_rays_{361};
  int default_hit_rank_{2};
  bool debug_{true};

  std::vector<WallSegment> walls_;
  std::vector<RsuConfig> rsus_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_publisher_;

  Point2D map_offset_{0.0, 0.0};
  bool is_offset_initialized_{false};

  void declare_and_get_params();
  void load_rsus_from_params();
  void load_walls_from_csv();
  void timer_callback();
  void publish_scan(const RsuConfig & rsu);
  void publish_debug_markers();

  bool wall_matches_rsu(const WallSegment & wall, const RsuConfig & rsu) const;
  std::optional<Point2D> get_line_segment_intersection(
    Point2D p1, Point2D p2, Point2D p3, Point2D p4) const;
  Point2D to_internal(Point2D point) const;
  Point2D to_map(Point2D point) const;
};

}  // namespace rsu_laserscan_generator

#endif  // RSU_LASERSCAN_GENERATOR_NODE_HPP_
