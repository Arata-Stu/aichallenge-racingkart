#ifndef TELEOP_MANAGER__JOY_INPUT_HPP_
#define TELEOP_MANAGER__JOY_INPUT_HPP_

#include <algorithm>

namespace teleop_manager
{

inline double trigger_press_amount(double axis_value)
{
  // DualShock triggers report +1.0 when released and -1.0 when fully pressed.
  return std::max(0.0, std::min(1.0, (1.0 - axis_value) * 0.5));
}

inline double signed_throttle(
  double positive_axis, double negative_axis, double deadzone)
{
  const double bounded_deadzone = std::max(0.0, std::min(1.0, deadzone));
  double positive = trigger_press_amount(positive_axis);
  double negative = trigger_press_amount(negative_axis);

  if (positive <= bounded_deadzone) {
    positive = 0.0;
  }
  if (negative <= bounded_deadzone) {
    negative = 0.0;
  }
  return positive - negative;
}

}  // namespace teleop_manager

#endif  // TELEOP_MANAGER__JOY_INPUT_HPP_
