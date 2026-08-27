#ifndef VIRTUAL_SCAN_BEV_GENERATOR__BEV_RASTERIZER_HPP_
#define VIRTUAL_SCAN_BEV_GENERATOR__BEV_RASTERIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace virtual_scan_bev_generator
{

enum class BevChannel : std::size_t
{
  kReachable = 0,
  kObservedFree = 1,
  kStaticWall = 2,
  kDynamicObstacle = 3,
  kObstacleOccluded = 4,
  kWallOccluded = 5,
  kRayCovered = 6,
  kEgo = 7,
};

constexpr std::size_t kBevChannelCount = 8;

struct GridConfig
{
  float x_min{-8.0F};
  float x_max{20.0F};
  float y_min{-16.0F};
  float y_max{16.0F};
  float resolution{0.2F};
  float dynamic_threshold{0.15F};
  float endpoint_radius{0.2F};
  float ego_radius{0.5F};
};

struct ScanGeometry
{
  float angle_min{0.0F};
  float angle_increment{0.0F};
  float range_min{0.0F};
  float range_max{0.0F};
  std::size_t ray_count{0};
};

class BevRasterizer
{
public:
  explicit BevRasterizer(const GridConfig & config);

  void configure(const ScanGeometry & geometry);
  void rasterize(
    const std::vector<float> & scan_with_obstacles,
    const std::vector<float> & scan_without_obstacles);

  [[nodiscard]] std::size_t width() const {return width_;}
  [[nodiscard]] std::size_t height() const {return height_;}
  [[nodiscard]] const GridConfig & grid_config() const {return config_;}
  [[nodiscard]] const ScanGeometry & scan_geometry() const {return geometry_;}
  [[nodiscard]] const std::vector<std::uint8_t> & data() const {return data_;}
  [[nodiscard]] std::uint8_t value(
    std::size_t x, std::size_t y, BevChannel channel) const;

  [[nodiscard]] bool world_to_cell(float x, float y, std::size_t & column, std::size_t & row) const;

private:
  struct RayCell
  {
    std::uint32_t index;
    float range;
  };

  GridConfig config_;
  ScanGeometry geometry_;
  std::size_t width_{0};
  std::size_t height_{0};
  std::vector<std::vector<RayCell>> ray_cells_;
  std::vector<std::uint8_t> data_;
  std::vector<std::array<int, 2>> endpoint_offsets_;
  std::vector<std::array<int, 2>> ego_offsets_;
  bool configured_{false};

  [[nodiscard]] std::size_t offset(std::size_t cell, BevChannel channel) const;
  void set(std::size_t cell, BevChannel channel);
  void stamp(float x, float y, BevChannel channel, const std::vector<std::array<int, 2>> & offsets);
  [[nodiscard]] std::vector<std::array<int, 2>> make_disk_offsets(float radius) const;
  [[nodiscard]] float sanitize_range(float range) const;
};

}  // namespace virtual_scan_bev_generator

#endif  // VIRTUAL_SCAN_BEV_GENERATOR__BEV_RASTERIZER_HPP_
