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
            "forward_overshoot": args.forward_overshoot_scale,
            "forward_wrong_direction": args.forward_wrong_direction_scale,
            "command_progress": args.command_progress_scale,
            "command_progress_shortfall": args.command_progress_shortfall_scale,
            "command_progress_failure": args.command_progress_failure_scale,
            "action_rate": args.action_rate_scale,
            "action_magnitude": args.action_magnitude_scale,
            "stand_still": args.stand_still_scale,
            "orientation": args.orientation_scale,
            "base_height": args.base_height_scale,
            "forward_pitch": args.forward_pitch_scale,
            "forward_pitch_rate": args.forward_pitch_rate_scale,
            "forward_contact_support": args.forward_contact_support_scale,
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
        if args.forward_overshoot_allowed_ratio is not None:
            config.reward_config.forward_overshoot_allowed_ratio = (
                args.forward_overshoot_allowed_ratio
            )
        if args.forward_wrong_direction_allowed_reverse_ratio is not None:
            config.reward_config.forward_wrong_direction_allowed_reverse_ratio = (
                args.forward_wrong_direction_allowed_reverse_ratio
            )
        if args.command_progress_required_ratio is not None:
            config.reward_config.command_progress_required_ratio = (
                args.command_progress_required_ratio
            )
        if args.command_progress_warmup_steps is not None:
            config.reward_config.command_progress_warmup_steps = (
                args.command_progress_warmup_steps
            )
        if args.command_progress_failure_enable:
            config.reward_config.command_progress_failure_enable = True
        if args.command_progress_failure_min_ratio is not None:
            config.reward_config.command_progress_failure_min_ratio = (
                args.command_progress_failure_min_ratio
            )
        if args.command_progress_failure_warmup_steps is not None:
            config.reward_config.command_progress_failure_warmup_steps = (
                args.command_progress_failure_warmup_steps
            )
        if args.reward_clip_min is not None:
            config.reward_config.reward_clip_min = args.reward_clip_min
        if args.reward_clip_max is not None:
            config.reward_config.reward_clip_max = args.reward_clip_max
        if args.forward_contact_support_no_contact_weight is not None:
            config.reward_config.forward_contact_support_no_contact_weight = (
                args.forward_contact_support_no_contact_weight
            )
        if args.forward_contact_support_asymmetry_weight is not None:
            config.reward_config.forward_contact_support_asymmetry_weight = (
                args.forward_contact_support_asymmetry_weight
            )
        reward_huber_overrides = {
            "action_rate_huber_delta": args.action_rate_huber_delta,
            "action_magnitude_huber_delta": args.action_magnitude_huber_delta,
            "target_rate_huber_delta": args.target_rate_huber_delta,
            "actuator_tracking_huber_delta": args.actuator_tracking_huber_delta,
            "forward_shortfall_huber_delta": args.forward_shortfall_huber_delta,
            "forward_overshoot_huber_delta": args.forward_overshoot_huber_delta,
            "forward_wrong_direction_huber_delta": (
                args.forward_wrong_direction_huber_delta
            ),
            "forward_pitch_huber_delta": args.forward_pitch_huber_delta,
            "forward_pitch_rate_huber_delta": args.forward_pitch_rate_huber_delta,
            "command_progress_shortfall_huber_delta": (
                args.command_progress_shortfall_huber_delta
            ),
        }
        for name, value in reward_huber_overrides.items():
            if value is not None:
                config.reward_config[name] = value

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
        "--forward_overshoot_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_overshoot. "
            "Use a negative value to penalize moving faster than the configured "
            "command ratio."
        ),
    )
    parser.add_argument(
        "--forward_overshoot_allowed_ratio",
        type=float,
        default=None,
        help="Optional allowed local forward speed ratio before overshoot cost starts.",
    )
    parser.add_argument(
        "--forward_wrong_direction_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_wrong_direction. "
            "Use a negative value to penalize reverse motion under a forward command."
        ),
    )
    parser.add_argument(
        "--forward_wrong_direction_allowed_reverse_ratio",
        type=float,
        default=None,
        help="Optional tolerated reverse-speed ratio before wrong-direction cost starts.",
    )
    parser.add_argument(
        "--command_progress_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.command_progress. "
            "Rewards cumulative signed forward progress over a command window."
        ),
    )
    parser.add_argument(
        "--command_progress_shortfall_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.command_progress_shortfall. "
            "Use a negative value to penalize low cumulative command progress."
        ),
    )
    parser.add_argument(
        "--command_progress_failure_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.command_progress_failure. "
            "Use a negative value with --reward_clip_min below zero to make "
            "command-progress termination carry a signed penalty."
        ),
    )
    parser.add_argument(
        "--command_progress_required_ratio",
        type=float,
        default=None,
        help="Optional required cumulative command progress ratio.",
    )
    parser.add_argument(
        "--command_progress_warmup_steps",
        type=int,
        default=None,
        help="Optional warmup steps before command_progress_shortfall is applied.",
    )
    parser.add_argument(
        "--command_progress_failure_enable",
        action="store_true",
        help=(
            "Terminate positive-command episodes after warmup when cumulative "
            "command progress remains below the configured floor. Default off."
        ),
    )
    parser.add_argument(
        "--command_progress_failure_min_ratio",
        type=float,
        default=None,
        help="Minimum cumulative command progress ratio before failure triggers.",
    )
    parser.add_argument(
        "--command_progress_failure_warmup_steps",
        type=int,
        default=None,
        help="Warmup steps before command-progress failure can terminate an episode.",
    )
    parser.add_argument(
        "--reward_clip_min",
        type=float,
        default=None,
        help="Optional lower bound for per-step clipped reward.",
    )
    parser.add_argument(
        "--reward_clip_max",
        type=float,
        default=None,
        help="Optional upper bound for per-step clipped reward.",
    )
    parser.add_argument(
        "--action_rate_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for action-rate cost. Default keeps "
            "the existing squared cost."
        ),
    )
    parser.add_argument(
        "--action_magnitude_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for action-magnitude cost. Default "
            "keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--target_rate_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for target-rate cost. Default keeps "
            "the existing squared cost."
        ),
    )
    parser.add_argument(
        "--actuator_tracking_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for actuator tracking cost. Default "
            "keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_shortfall_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward shortfall cost. Default "
            "keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_overshoot_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward overshoot cost. Default "
            "keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_wrong_direction_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for wrong-direction cost. Default "
            "keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_pitch_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward-command pitch cost. "
            "Default keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_pitch_rate_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward-command pitch-rate cost. "
            "Default keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--command_progress_shortfall_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for command-window shortfall cost. "
            "Default keeps the existing squared cost."
        ),
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
        "--orientation_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.orientation.",
    )
    parser.add_argument(
        "--base_height_scale",
        type=float,
        default=None,
        help="Optional override for reward_config.scales.base_height.",
    )
    parser.add_argument(
        "--forward_pitch_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_pitch. Use a "
            "negative value to penalize forward-command pitch tilt."
        ),
    )
    parser.add_argument(
        "--forward_pitch_rate_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_pitch_rate. Use "
            "a negative value to penalize forward-command pitch-rate motion."
        ),
    )
    parser.add_argument(
        "--forward_contact_support_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_contact_support. "
            "Use a negative value to penalize unsupported contacts under a "
            "forward command."
        ),
    )
    parser.add_argument(
        "--forward_contact_support_no_contact_weight",
        type=float,
        default=None,
        help="Optional weight for no-foot-contact support cost.",
    )
    parser.add_argument(
        "--forward_contact_support_asymmetry_weight",
        type=float,
        default=None,
        help="Optional weight for one-sided support cost.",
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
    parser.add_argument("--ppo_learning_rate", type=float, default=None)
    parser.add_argument("--ppo_entropy_cost", type=float, default=None)
    parser.add_argument("--ppo_clipping_epsilon", type=float, default=None)
    parser.add_argument("--ppo_max_grad_norm", type=float, default=None)
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
