"""Training, resume, checking, and evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Virtual Scan SAC for AWSIM")
    parser.add_argument("--config", required=True, help="lap.yaml or overtake.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Run Gymnasium environment checks")
    train = subparsers.add_parser("train", help="Start a new SAC run")
    train.add_argument("--timesteps", type=int, default=None)
    resume = subparsers.add_parser("resume", help="Resume a SAC checkpoint")
    resume.add_argument("--model", required=True)
    resume.add_argument("--replay-buffer", default=None)
    resume.add_argument("--timesteps", type=int, default=None)
    evaluate = subparsers.add_parser("evaluate", help="Run deterministic policy episodes")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=int, default=5)
    return parser


def _build_model(config: dict, env):
    from .model import TinyLidarFeatureExtractor
    from .sac import InterventionSAC

    sac = config["sac"]
    policy_kwargs = {
        "features_extractor_class": TinyLidarFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": int(sac["features_dim"])},
        "net_arch": {"pi": list(sac["net_arch"]), "qf": list(sac["net_arch"])},
    }
    return InterventionSAC(
        "MultiInputPolicy",
        env,
        learning_rate=float(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        tau=float(sac["tau"]),
        gamma=float(sac["gamma"]),
        train_freq=int(sac["train_freq"]),
        gradient_steps=int(sac["gradient_steps"]),
        ent_coef=sac["ent_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=config["output"]["tensorboard_dir"],
        device=str(sac["device"]),
        seed=int(sac["seed"]),
        verbose=1,
    )


def _learn(model, config: dict, timesteps: int | None, reset_num_timesteps: bool) -> None:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from .callbacks import DrivingMetricsCallback

    output = Path(config["output"]["checkpoint_dir"])
    output.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(
        save_freq=int(config["sac"]["checkpoint_freq"]),
        save_path=str(output),
        name_prefix=f"{config['stage']}_sac",
        save_replay_buffer=bool(config["sac"]["save_replay_buffer"]),
    )
    steps = int(timesteps or config["sac"]["total_timesteps"])
    callbacks = CallbackList([callback, DrivingMetricsCallback()])
    model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=reset_num_timesteps)
    model.save(str(output / "last_model"))
    if bool(config["sac"]["save_replay_buffer"]):
        model.save_replay_buffer(str(output / "last_replay_buffer.pkl"))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    from .config import load_config
    from .env import VirtualScanAWSIMEnv
    from .sac import InterventionSAC

    config = load_config(args.config)
    raw_env = VirtualScanAWSIMEnv(config)
    env = Monitor(raw_env)
    try:
        if args.command == "check":
            check_env(raw_env, warn=True)
            print("Environment check passed")
        elif args.command == "train":
            _learn(_build_model(config, env), config, args.timesteps, True)
        elif args.command == "resume":
            model = InterventionSAC.load(args.model, env=env, device=config["sac"]["device"])
            if args.replay_buffer:
                model.load_replay_buffer(args.replay_buffer)
            _learn(model, config, args.timesteps, False)
        elif args.command == "evaluate":
            model = InterventionSAC.load(args.model, env=env, device=config["sac"]["device"])
            for episode in range(args.episodes):
                observation, _ = env.reset()
                done = False
                total_reward = 0.0
                info = {}
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated
                print(
                    f"episode={episode + 1} reward={total_reward:.2f} "
                    f"reason={info.get('termination_reason')} "
                    f"lap={info.get('lap_count')} lap_time={info.get('lap_time_s', 0.0):.2f}s "
                    f"progress={info.get('total_progress_m', 0.0):.1f}m"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
