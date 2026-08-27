"""Dependency-light public interfaces for the Step 2 racing curriculum."""

from lidar_racing_rl.curriculum.opponent_pool import (
    OPPONENT_POOL_SCHEMA_VERSION,
    SAMPLING_ALGORITHM,
    OpponentCheckpoint,
    OpponentPool,
    OpponentPoolConfigurationError,
    OpponentPoolNotIntegratedError,
    OpponentPoolSettings,
    PoolValidationPolicy,
    deterministic_pool_sample,
    load_configured_opponent_pool,
    load_opponent_pool_manifest,
    verify_opponent_checkpoint,
)
from lidar_racing_rl.curriculum.schedule import (
    CURRICULUM_SCHEMA_VERSION,
    STEP2_PHASES,
    CurriculumConfigurationError,
    CurriculumPhase,
    CurriculumPlan,
    UnsupportedCurriculumPhaseError,
)


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "OPPONENT_POOL_SCHEMA_VERSION",
    "SAMPLING_ALGORITHM",
    "STEP2_PHASES",
    "CurriculumConfigurationError",
    "CurriculumPhase",
    "CurriculumPlan",
    "OpponentCheckpoint",
    "OpponentPool",
    "OpponentPoolConfigurationError",
    "OpponentPoolNotIntegratedError",
    "OpponentPoolSettings",
    "PoolValidationPolicy",
    "UnsupportedCurriculumPhaseError",
    "deterministic_pool_sample",
    "load_configured_opponent_pool",
    "load_opponent_pool_manifest",
    "verify_opponent_checkpoint",
]
