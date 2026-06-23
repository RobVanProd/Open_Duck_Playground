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
        reward_scale_overrides = {
            "tracking_lin_vel": args.tracking_lin_vel_scale,
            "tracking_ang_vel": args.tracking_ang_vel_scale,
            "forward_progress": args.forward_progress_scale,
            "forward_shortfall": args.forward_shortfall_scale,
            "action_rate": args.action_rate_scale,
            "action_magnitude": args.action_magnitude_scale,
            "stand_still": args.stand_still_scale,
            "alive": args.alive_scale,
            "imitation": args.imitation_scale,
        }
        for name, value in reward_scale_overrides.items():
            if value is not None:
                config.reward_config.scales[name] = value

        if args.tracking_sigma is not None:
            config.reward_config.tracking_sigma = args.tracking_sigma
        if args.forward_progress_deadband is not None:
            config.reward_config.forward_progress_deadband = (
                args.forward_progress_deadband
            )
        if args.forward_shortfall_required_ratio is not None:
            config.reward_config.forward_shortfall_required_ratio = (
                args.forward_shortfall_required_ratio
            )

        command_range_overrides = {
            "lin_vel_x": (args.lin_vel_x_min, args.lin_vel_x_max),
            "lin_vel_y": (args.lin_vel_y_min, args.lin_vel_y_max),
            "ang_vel_yaw": (args.ang_vel_yaw_min, args.ang_vel_yaw_max),
        }
        for name, (min_value, max_value) in command_range_overrides.items():
            if min_value is not None or max_value is not None:
                current_min, current_max = config[name]
                config[name] = [
                    current_min if min_value is None else min_value,
                    current_max if max_value is None else max_value,
                ]

        if args.head_range_factor is not None:
            config.head_range_factor = args.head_range_factor
        if args.command_resample_steps is not None:
            config.command_resample_steps = args.command_resample_steps
        if args.zero_command_probability is not None:
            config.zero_command_probability = args.zero_command_probability
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
        help=(
            "Scale applied to the target velocity cost. Use a negative value "
            "to penalize the cost; positive values reward it. Default keeps "
            "behavior unchanged."
        ),
    )
    parser.add_argument(
        "--actuator_tracking_scale",
        type=float,
        default=0.0,
        help=(
            "Scale applied to the sent-vs-applied target cost. Use a negative "
            "value to penalize the cost; positive values reward it. Default "
            "keeps behavior unchanged."
        ),
    )
    parser.add_argument(
        "--tracking_lin_vel_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.tracking_lin_vel.",
    )
    parser.add_argument(
        "--tracking_ang_vel_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.tracking_ang_vel.",
    )
    parser.add_argument(
        "--tracking_sigma",
        type=float,
        default=None,
        help="Optional override for reward_config.tracking_sigma.",
    )
    parser.add_argument(
        "--forward_progress_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.forward_progress.",
    )
    parser.add_argument(
        "--forward_shortfall_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_shortfall. "
            "Use a negative value to penalize failing to reach the configured "
            "fraction of commanded forward speed."
        ),
    )
    parser.add_argument(
        "--forward_progress_deadband",
        type=float,
        default=None,
        help="Optional override for reward_config.forward_progress_deadband.",
    )
    parser.add_argument(
        "--forward_shortfall_required_ratio",
        type=float,
        default=None,
        help="Optional required fraction of command_x for forward_shortfall.",
    )
    parser.add_argument(
        "--action_rate_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.action_rate.",
    )
    parser.add_argument(
        "--action_magnitude_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.action_magnitude.",
    )
    parser.add_argument(
        "--stand_still_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.stand_still.",
    )
    parser.add_argument(
        "--alive_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.alive.",
    )
    parser.add_argument(
        "--imitation_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.imitation.",
    )
    parser.add_argument("--lin_vel_x_min", type=float, default=None)
    parser.add_argument("--lin_vel_x_max", type=float, default=None)
    parser.add_argument("--lin_vel_y_min", type=float, default=None)
    parser.add_argument("--lin_vel_y_max", type=float, default=None)
    parser.add_argument("--ang_vel_yaw_min", type=float, default=None)
    parser.add_argument("--ang_vel_yaw_max", type=float, default=None)
    parser.add_argument(
        "--command_resample_steps",
        type=int,
        default=None,
        help="Optional override for command resampling interval in env steps.",
    )
    parser.add_argument(
        "--zero_command_probability",
        type=float,
        default=None,
        help="Optional override for probability of sampling an all-zero command.",
    )
    parser.add_argument(
        "--head_range_factor",
        type=float,
        default=None,
        help="Optional override for sampled head command range multiplier.",
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
