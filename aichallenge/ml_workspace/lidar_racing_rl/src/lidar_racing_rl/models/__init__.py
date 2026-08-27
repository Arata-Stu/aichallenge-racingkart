"""Training and deployment model modules.

The initializer deliberately performs no eager imports: the PyTorch-only ROS
deployment path must not acquire a JAX/Flax dependency merely by importing
``lidar_racing_rl.models.actor_torch``.  Import each backend from its defining
module.
"""
