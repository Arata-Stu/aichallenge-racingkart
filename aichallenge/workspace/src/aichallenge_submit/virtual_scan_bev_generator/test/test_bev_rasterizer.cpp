#include "virtual_scan_bev_generator/bev_rasterizer.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace virtual_scan_bev_generator
{

namespace
{
std::uint8_t at_world(
  const BevRasterizer & rasterizer, const float x, const float y, const BevChannel channel)
{
  std::size_t column = 0U;
  std::size_t row = 0U;
  EXPECT_TRUE(rasterizer.world_to_cell(x, y, column, row));
  return rasterizer.value(column, row, channel);
}
}  // namespace

TEST(BevRasterizer, SeparatesDynamicObstacleFromStaticWall)
{
  GridConfig config;
  config.x_min = -1.0F;
  config.x_max = 7.0F;
  config.y_min = -2.0F;
  config.y_max = 2.0F;
  config.resolution = 0.1F;
  config.dynamic_threshold = 0.15F;
  config.endpoint_radius = 0.0F;
  config.ego_radius = 0.0F;
  BevRasterizer rasterizer(config);
  rasterizer.configure(ScanGeometry{0.0F, 0.1F, 0.01F, 6.0F, 1U});

  rasterizer.rasterize({2.0F}, {4.0F});

  EXPECT_EQ(at_world(rasterizer, 1.0F, 0.0F, BevChannel::kObservedFree), 255U);
  EXPECT_EQ(at_world(rasterizer, 2.0F, 0.0F, BevChannel::kDynamicObstacle), 255U);
  EXPECT_EQ(at_world(rasterizer, 3.0F, 0.0F, BevChannel::kObstacleOccluded), 255U);
  EXPECT_EQ(at_world(rasterizer, 4.0F, 0.0F, BevChannel::kStaticWall), 255U);
  EXPECT_EQ(at_world(rasterizer, 5.0F, 0.0F, BevChannel::kWallOccluded), 255U);
  EXPECT_EQ(at_world(rasterizer, 5.0F, 0.0F, BevChannel::kReachable), 0U);
}

TEST(BevRasterizer, DoesNotInventDynamicObstacleForSmallRangeNoise)
{
  GridConfig config;
  config.x_min = -1.0F;
  config.x_max = 7.0F;
  config.y_min = -2.0F;
  config.y_max = 2.0F;
  config.resolution = 0.1F;
  config.dynamic_threshold = 0.15F;
  config.endpoint_radius = 0.0F;
  config.ego_radius = 0.0F;
  BevRasterizer rasterizer(config);
  rasterizer.configure(ScanGeometry{0.0F, 0.1F, 0.01F, 6.0F, 1U});

  rasterizer.rasterize({3.90F}, {4.0F});

  EXPECT_EQ(at_world(rasterizer, 3.9F, 0.0F, BevChannel::kDynamicObstacle), 0U);
  EXPECT_EQ(at_world(rasterizer, 4.0F, 0.0F, BevChannel::kStaticWall), 255U);
}

TEST(BevRasterizer, InfiniteStaticRangeLeavesRayReachable)
{
  GridConfig config;
  config.x_min = -1.0F;
  config.x_max = 5.0F;
  config.y_min = -1.0F;
  config.y_max = 1.0F;
  config.resolution = 0.1F;
  config.endpoint_radius = 0.0F;
  config.ego_radius = 0.0F;
  BevRasterizer rasterizer(config);
  rasterizer.configure(ScanGeometry{0.0F, 0.1F, 0.01F, 4.0F, 1U});

  const float infinity = std::numeric_limits<float>::infinity();
  rasterizer.rasterize({infinity}, {infinity});

  EXPECT_EQ(at_world(rasterizer, 3.0F, 0.0F, BevChannel::kReachable), 255U);
  EXPECT_EQ(at_world(rasterizer, 3.0F, 0.0F, BevChannel::kWallOccluded), 0U);
}

TEST(BevRasterizer, TypicalFrameIsFastEnoughForOnlineGeneration)
{
  GridConfig config;
  BevRasterizer rasterizer(config);
  constexpr std::size_t ray_count = 1080U;
  constexpr float pi = 3.14159265358979323846F;
  rasterizer.configure(ScanGeometry{
      -0.75F * pi, 1.5F * pi / static_cast<float>(ray_count - 1U),
      0.001F, 30.0F, ray_count});

  std::vector<float> static_scan(ray_count, 12.0F);
  std::vector<float> obstacle_scan = static_scan;
  for (std::size_t ray = 480U; ray < 600U; ++ray) {
    obstacle_scan[ray] = 6.0F;
  }

  constexpr int iterations = 100;
  const auto started = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    rasterizer.rasterize(obstacle_scan, static_scan);
  }
  const double average_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count() / iterations;

  EXPECT_EQ(rasterizer.width(), 140U);
  EXPECT_EQ(rasterizer.height(), 160U);
  EXPECT_LT(average_ms, 20.0) << "Average rasterization took " << average_ms << " ms";
}

}  // namespace virtual_scan_bev_generator
