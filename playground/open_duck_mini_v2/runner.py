"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self._build_env_config(args)
        self.env = self.env_file[1](task=args.task, config=self.env_config)
        self.eval_env = self.env_file[1](task=args.task, config=self.env_config)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

    def _build_env_config(self, args):
        config = self.env_file[0].default_config()

        if args.enable_actuator_bridge:
            if not hasattr(config, "actuator_bridge"):
                raise ValueError("Selected env config does not expose actuator_bridge")
            config.actuator_bridge.enable = True
            config.actuator_bridge.delay_min_ticks = args.actuator_bridge_delay_min_ticks
            config.actuator_bridge.delay_max_ticks = args.actuator_bridge_delay_max_ticks
            config.actuator_bridge.tau_min_s = args.actuator_bridge_tau_min_s
            config.actuator_bridge.tau_max_s = args.actuator_bridge_tau_max_s
            config.actuator_bridge.velocity_limit_min_rad_s = (
                args.actuator_bridge_velocity_limit_min_rad_s
            )
            config.actuator_bridge.velocity_limit_max_rad_s = (
                args.actuator_bridge_velocity_limit_max_rad_s
            )
            config.actuator_bridge.per_joint_variation = (
                args.actuator_bridge_per_joint_variation
            )

        config.reward_config.scales.target_rate = args.target_rate_scale
        config.reward_config.scales.actuator_tracking = args.actuator_tracking_scale
        return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--enable_actuator_bridge",
        action="store_true",
        help="Enable the default-off actuator delay/lag/velocity-limit bridge",
    )
    parser.add_argument("--actuator_bridge_delay_min_ticks", type=int, default=3)
    parser.add_argument("--actuator_bridge_delay_max_ticks", type=int, default=8)
    parser.add_argument("--actuator_bridge_tau_min_s", type=float, default=0.06)
    parser.add_argument("--actuator_bridge_tau_max_s", type=float, default=0.14)
    parser.add_argument(
        "--actuator_bridge_velocity_limit_min_rad_s", type=float, default=2.5
    )
    parser.add_argument(
        "--actuator_bridge_velocity_limit_max_rad_s", type=float, default=4.7
    )
    parser.add_argument("--actuator_bridge_per_joint_variation", type=float, default=0.15)
    parser.add_argument(
        "--target_rate_scale",
        type=float,
        default=0.0,
        help="Reward scale for target velocity cost; default keeps behavior unchanged",
    )
    parser.add_argument(
        "--actuator_tracking_scale",
        type=float,
        default=0.0,
        help="Reward scale for sent-vs-applied target cost; default keeps behavior unchanged",
    )
    parser.add_argument("--ppo_num_envs", type=int, default=None)
    parser.add_argument("--ppo_num_evals", type=int, default=None)
    parser.add_argument("--ppo_episode_length", type=int, default=None)
    parser.add_argument("--ppo_unroll_length", type=int, default=None)
    parser.add_argument("--ppo_batch_size", type=int, default=None)
    parser.add_argument("--ppo_num_minibatches", type=int, default=None)
    parser.add_argument("--ppo_num_updates_per_batch", type=int, default=None)
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
