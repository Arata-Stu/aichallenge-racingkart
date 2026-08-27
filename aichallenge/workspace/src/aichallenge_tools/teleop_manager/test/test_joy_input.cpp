#include <gtest/gtest.h>

#include "teleop_manager/joy_input.hpp"

TEST(JoyInput, TriggerPressAmountUsesDualShockRange)
{
  EXPECT_DOUBLE_EQ(teleop_manager::trigger_press_amount(1.0), 0.0);
  EXPECT_DOUBLE_EQ(teleop_manager::trigger_press_amount(0.0), 0.5);
  EXPECT_DOUBLE_EQ(teleop_manager::trigger_press_amount(-1.0), 1.0);
}

TEST(JoyInput, TriggerPressAmountClampsUnexpectedValues)
{
  EXPECT_DOUBLE_EQ(teleop_manager::trigger_press_amount(2.0), 0.0);
  EXPECT_DOUBLE_EQ(teleop_manager::trigger_press_amount(-2.0), 1.0);
}

TEST(JoyInput, R2ProducesPositiveThrottle)
{
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(-1.0, 1.0, 0.05), 1.0);
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(0.0, 1.0, 0.05), 0.5);
}

TEST(JoyInput, L2ProducesNegativeThrottle)
{
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(1.0, -1.0, 0.05), -1.0);
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(1.0, 0.0, 0.05), -0.5);
}

TEST(JoyInput, SimultaneousTriggerPressesCancel)
{
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(-1.0, -1.0, 0.05), 0.0);
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(0.0, 0.0, 0.05), 0.0);
}

TEST(JoyInput, DeadzoneSuppressesNeutralNoise)
{
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(0.92, 1.0, 0.05), 0.0);
  EXPECT_DOUBLE_EQ(teleop_manager::signed_throttle(1.0, 0.92, 0.05), 0.0);
}
