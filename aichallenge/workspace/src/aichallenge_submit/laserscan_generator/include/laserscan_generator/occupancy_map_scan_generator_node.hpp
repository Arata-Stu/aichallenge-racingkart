#ifndef OCCUPANCY_MAP_SCAN_GENERATOR_NODE_HPP_
#define OCCUPANCY_MAP_SCAN_GENERATOR_NODE_HPP_

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

class OccupancyMapScanGeneratorNode : public rclcpp::Node
{
public:
  OccupancyMapScanGeneratorNode();

private:
  struct PixelIndex
  {
    int x;
    int y;
  };

  void declare_and_get_params();
  bool validate_params() const;
  bool load_map();
  void timer_callback();
  float raycast(double origin_x, double origin_y, double yaw, double relative_angle) const;
  bool world_to_pixel(double world_x, double world_y, PixelIndex & pixel) const;
  bool is_occupied(const PixelIndex & pixel) const;
  void publish_debug_points(
    const std::vector<geometry_msgs::msg::Point> & hit_points, const rclcpp::Time & stamp);

  std::string map_path_;
  std::string map_frame_id_;
  std::string scan_frame_id_;
  std::string output_topic_;

  double resolution_;
  double origin_x_;
  double origin_y_;
  bool invert_y_axis_;

  double timer_hz_;
  double angle_min_deg_;
  double angle_max_deg_;
  int num_rays_;
  double range_min_;
  double range_max_;
  double ray_step_;
  int occupied_threshold_;
  bool debug_;

  cv::Mat map_gray_;
  std::vector<double> relative_angles_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr debug_hit_points_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif  // OCCUPANCY_MAP_SCAN_GENERATOR_NODE_HPP_
