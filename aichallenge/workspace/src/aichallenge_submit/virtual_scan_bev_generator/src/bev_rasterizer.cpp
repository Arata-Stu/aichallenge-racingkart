#include "virtual_scan_bev_generator/bev_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace virtual_scan_bev_generator
{

namespace
{
constexpr std::uint8_t kSet = 255U;
}

BevRasterizer::BevRasterizer(const GridConfig & config)
: config_(config)
{
  if (!(config_.resolution > 0.0F) || !(config_.x_max > config_.x_min) ||
    !(config_.y_max > config_.y_min))
  {
    throw std::invalid_argument("BEV grid bounds and resolution must be positive");
  }
  if (config_.dynamic_threshold < 0.0F || config_.endpoint_radius < 0.0F ||
    config_.ego_radius < 0.0F)
  {
    throw std::invalid_argument("BEV distances must be non-negative");
  }

  width_ = static_cast<std::size_t>(
    std::ceil((config_.x_max - config_.x_min) / config_.resolution));
  height_ = static_cast<std::size_t>(
    std::ceil((config_.y_max - config_.y_min) / config_.resolution));
  data_.resize(width_ * height_ * kBevChannelCount, 0U);
  endpoint_offsets_ = make_disk_offsets(config_.endpoint_radius);
  ego_offsets_ = make_disk_offsets(config_.ego_radius);
}

void BevRasterizer::configure(const ScanGeometry & geometry)
{
  if (geometry.ray_count == 0U || !(geometry.angle_increment > 0.0F) ||
    !(geometry.range_max > geometry.range_min))
  {
    throw std::invalid_argument("Invalid LaserScan geometry");
  }

  geometry_ = geometry;
  ray_cells_.clear();
  ray_cells_.resize(geometry_.ray_count);

  // Half-cell sampling avoids gaps between rays while consecutive duplicate
  // cells are discarded. This lookup is reused for every scan frame.
  const float step = config_.resolution * 0.5F;
  for (std::size_t ray = 0; ray < geometry_.ray_count; ++ray) {
    const float angle = geometry_.angle_min + static_cast<float>(ray) * geometry_.angle_increment;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    std::uint32_t previous = std::numeric_limits<std::uint32_t>::max();
    auto & cells = ray_cells_[ray];

    for (float range = 0.0F; range <= geometry_.range_max; range += step) {
      std::size_t column = 0U;
      std::size_t row = 0U;
      if (!world_to_cell(range * cosine, range * sine, column, row)) {
        // A ray can only leave this convex rectangular grid once.
        if (range > 0.0F) {
          break;
        }
        continue;
      }
      const auto index = static_cast<std::uint32_t>(row * width_ + column);
      if (index != previous) {
        cells.push_back({index, range});
        previous = index;
      }
    }
  }
  configured_ = true;
}

void BevRasterizer::rasterize(
  const std::vector<float> & scan_with_obstacles,
  const std::vector<float> & scan_without_obstacles)
{
  if (!configured_) {
    throw std::logic_error("Rasterizer must be configured before rasterize()");
  }
  if (scan_with_obstacles.size() != geometry_.ray_count ||
    scan_without_obstacles.size() != geometry_.ray_count)
  {
    throw std::invalid_argument("LaserScan ray count does not match configured geometry");
  }

  std::fill(data_.begin(), data_.end(), 0U);

  for (std::size_t ray = 0; ray < geometry_.ray_count; ++ray) {
    const float static_range = sanitize_range(scan_without_obstacles[ray]);
    const float observed_range = sanitize_range(scan_with_obstacles[ray]);
    const bool has_wall = std::isfinite(static_range);
    const bool has_observed_hit = std::isfinite(observed_range);
    const bool has_dynamic = has_observed_hit &&
      (!has_wall || observed_range + config_.dynamic_threshold < static_range);
    const float free_limit = has_dynamic ? observed_range :
      (has_wall ? static_range : geometry_.range_max);
    const float reachable_limit = has_wall ? static_range : geometry_.range_max;

    for (const auto & sample : ray_cells_[ray]) {
      set(sample.index, BevChannel::kRayCovered);
      if (sample.range <= reachable_limit) {
        set(sample.index, BevChannel::kReachable);
      }
      if (sample.range + config_.resolution * 0.5F < free_limit) {
        set(sample.index, BevChannel::kObservedFree);
      } else if (has_dynamic && sample.range > observed_range &&
        sample.range < reachable_limit)
      {
        set(sample.index, BevChannel::kObstacleOccluded);
      }
    }

    const float angle = geometry_.angle_min + static_cast<float>(ray) * geometry_.angle_increment;
    if (has_wall) {
      stamp(
        static_range * std::cos(angle), static_range * std::sin(angle),
        BevChannel::kStaticWall, endpoint_offsets_);
    }
    if (has_dynamic) {
      stamp(
        observed_range * std::cos(angle), observed_range * std::sin(angle),
        BevChannel::kDynamicObstacle, endpoint_offsets_);
    }
  }

  const std::size_t cell_count = width_ * height_;
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    const bool covered = data_[offset(cell, BevChannel::kRayCovered)] != 0U;
    const bool reachable = data_[offset(cell, BevChannel::kReachable)] != 0U;
    const bool free = data_[offset(cell, BevChannel::kObservedFree)] != 0U;
    if (covered && !reachable) {
      set(cell, BevChannel::kWallOccluded);
    }
    if (free || !reachable) {
      data_[offset(cell, BevChannel::kObstacleOccluded)] = 0U;
    }
  }

  stamp(0.0F, 0.0F, BevChannel::kEgo, ego_offsets_);
}

std::uint8_t BevRasterizer::value(
  const std::size_t x, const std::size_t y, const BevChannel channel) const
{
  if (x >= width_ || y >= height_) {
    throw std::out_of_range("BEV cell is outside the grid");
  }
  return data_[offset(y * width_ + x, channel)];
}

bool BevRasterizer::world_to_cell(
  const float x, const float y, std::size_t & column, std::size_t & row) const
{
  if (x < config_.x_min || x >= config_.x_max || y < config_.y_min || y >= config_.y_max) {
    return false;
  }
  column = static_cast<std::size_t>((x - config_.x_min) / config_.resolution);
  row = static_cast<std::size_t>((y - config_.y_min) / config_.resolution);
  return column < width_ && row < height_;
}

std::size_t BevRasterizer::offset(const std::size_t cell, const BevChannel channel) const
{
  return cell * kBevChannelCount + static_cast<std::size_t>(channel);
}

void BevRasterizer::set(const std::size_t cell, const BevChannel channel)
{
  data_[offset(cell, channel)] = kSet;
}

void BevRasterizer::stamp(
  const float x, const float y, const BevChannel channel,
  const std::vector<std::array<int, 2>> & offsets)
{
  std::size_t center_x = 0U;
  std::size_t center_y = 0U;
  if (!world_to_cell(x, y, center_x, center_y)) {
    return;
  }
  for (const auto & delta : offsets) {
    const int column = static_cast<int>(center_x) + delta[0];
    const int row = static_cast<int>(center_y) + delta[1];
    if (column >= 0 && row >= 0 && column < static_cast<int>(width_) &&
      row < static_cast<int>(height_))
    {
      set(
        static_cast<std::size_t>(row) * width_ + static_cast<std::size_t>(column), channel);
    }
  }
}

std::vector<std::array<int, 2>> BevRasterizer::make_disk_offsets(const float radius) const
{
  const int cells = static_cast<int>(std::ceil(radius / config_.resolution));
  std::vector<std::array<int, 2>> offsets;
  for (int y = -cells; y <= cells; ++y) {
    for (int x = -cells; x <= cells; ++x) {
      const float distance = std::hypot(
        static_cast<float>(x) * config_.resolution,
        static_cast<float>(y) * config_.resolution);
      if (distance <= radius + config_.resolution * 0.25F) {
        offsets.push_back({x, y});
      }
    }
  }
  if (offsets.empty()) {
    offsets.push_back({0, 0});
  }
  return offsets;
}

float BevRasterizer::sanitize_range(const float range) const
{
  if (!std::isfinite(range) || range < geometry_.range_min || range > geometry_.range_max) {
    return std::numeric_limits<float>::infinity();
  }
  return range;
}

}  // namespace virtual_scan_bev_generator
