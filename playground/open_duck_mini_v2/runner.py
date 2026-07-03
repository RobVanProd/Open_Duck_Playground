"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import json
from pathlib import Path

import numpy as np

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
        self.randomizer = randomize.make_domain_randomizer(
            self._build_domain_randomization_config(args)
        )
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

    def _build_domain_randomization_config(self, args):
        return {
            "friction_min": args.dr_friction_min,
            "friction_max": args.dr_friction_max,
            "frictionloss_scale_min": args.dr_frictionloss_scale_min,
            "frictionloss_scale_max": args.dr_frictionloss_scale_max,
            "armature_scale_min": args.dr_armature_scale_min,
            "armature_scale_max": args.dr_armature_scale_max,
            "com_jitter_m": args.dr_com_jitter_m,
            "mass_scale_min": args.dr_mass_scale_min,
            "mass_scale_max": args.dr_mass_scale_max,
            "torso_mass_delta_min": args.dr_torso_mass_delta_min,
            "torso_mass_delta_max": args.dr_torso_mass_delta_max,
            "qpos_jitter_rad": args.dr_qpos_jitter_rad,
            "actuator_gain_scale_min": args.dr_actuator_gain_scale_min,
            "actuator_gain_scale_max": args.dr_actuator_gain_scale_max,
            "leg_geometry_jitter_scale": args.dr_leg_geometry_jitter_scale,
        }

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

        if args.enable_soft_prior:
            if not hasattr(config, "soft_prior"):
                raise ValueError("Selected env config does not expose soft_prior")
            config.soft_prior.enable = True
            config.soft_prior.phase_source = args.soft_prior_phase_source
            config.soft_prior.huber_delta = args.soft_prior_huber_delta
            if args.soft_prior_config_json is None:
                raise ValueError("--enable_soft_prior requires --soft_prior_config_json")
            payload = json.loads(Path(args.soft_prior_config_json).read_text())
            prior = payload.get("prior") or {}
            action_mean = prior.get("action_mean") or []
            joint_indices = prior.get("joint_indices") or []
            if not action_mean or not joint_indices:
                raise ValueError("soft-prior config missing prior.action_mean/joint_indices")
            config.soft_prior.action_mean = action_mean
            config.soft_prior.joint_indices = joint_indices
            config.soft_prior.period = int(prior.get("window_len") or len(action_mean))
            config.reward_config.scales.soft_prior = args.soft_prior_scale

        if args.enable_behavior_prior:
            if not hasattr(config, "behavior_prior"):
                raise ValueError("Selected env config does not expose behavior_prior")
            if args.behavior_prior_mlp_npz is None:
                raise ValueError("--enable_behavior_prior requires --behavior_prior_mlp_npz")
            payload = np.load(Path(args.behavior_prior_mlp_npz), allow_pickle=False)
            if "norm" not in payload.files:
                raise ValueError("behavior-prior NPZ missing norm array")
            norm = payload["norm"].astype(np.float32)
            if norm.shape[0] != 2:
                raise ValueError(f"behavior-prior norm must have shape [2, obs], got {norm.shape}")
            weights = []
            biases = []
            index = 0
            while f"w{index}" in payload.files:
                weights.append(payload[f"w{index}"].astype(np.float32).tolist())
                biases.append(payload[f"b{index}"].astype(np.float32).tolist())
                index += 1
            if not weights:
                raise ValueError("behavior-prior NPZ contains no w*/b* layers")
            activation = str(payload["activation"][0]) if "activation" in payload.files else "tanh"
            output_mode = str(payload["output_mode"][0]) if "output_mode" in payload.files else "clip"
            if activation not in {"tanh", "swish"}:
                raise ValueError(f"unsupported behavior-prior activation {activation!r}")
            if output_mode not in {"clip", "ppo_tanh_loc"}:
                raise ValueError(f"unsupported behavior-prior output_mode {output_mode!r}")
            config.behavior_prior.enable = True
            config.behavior_prior.obs_mean = norm[0].tolist()
            config.behavior_prior.obs_std = norm[1].tolist()
            config.behavior_prior.weights = weights
            config.behavior_prior.biases = biases
            config.behavior_prior.activation = activation
            config.behavior_prior.output_mode = output_mode
            config.behavior_prior.huber_delta = args.behavior_prior_huber_delta
            config.reward_config.scales.behavior_prior = args.behavior_prior_scale

        config.reward_config.scales.target_rate = args.target_rate_scale
        config.reward_config.scales.actuator_tracking = args.actuator_tracking_scale
        reward_scale_overrides = {
            "push_recovery_actuator_tracking": (
                args.push_recovery_actuator_tracking_scale
            ),
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
            "forward_single_support": args.forward_single_support_scale,
            "forward_double_support": args.forward_double_support_scale,
            "forward_contact_transition": args.forward_contact_transition_scale,
            "forward_double_support_dwell": args.forward_double_support_dwell_scale,
            "forward_swing_clearance": args.forward_swing_clearance_scale,
            "forward_phase_swing_lift": args.forward_phase_swing_lift_scale,
            "forward_phase_single_support": (
                args.forward_phase_single_support_scale
            ),
            "forward_swing_balance": args.forward_swing_balance_scale,
            "forward_swing_advance": args.forward_swing_advance_scale,
            "forward_swing_target_rate_limit": (
                args.forward_swing_target_rate_limit_scale
            ),
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
        if args.forward_contact_transition_min_progress_ratio is not None:
            config.reward_config.forward_contact_transition_min_progress_ratio = (
                args.forward_contact_transition_min_progress_ratio
            )
        if args.forward_double_support_dwell_grace_steps is not None:
            config.reward_config.forward_double_support_dwell_grace_steps = (
                args.forward_double_support_dwell_grace_steps
            )
        if args.forward_swing_clearance_target_m is not None:
            config.reward_config.forward_swing_clearance_target_m = (
                args.forward_swing_clearance_target_m
            )
        if args.forward_phase_swing_lift_target_m is not None:
            config.reward_config.forward_phase_swing_lift_target_m = (
                args.forward_phase_swing_lift_target_m
            )
        if args.forward_phase_single_support_swing_contact_weight is not None:
            config.reward_config.forward_phase_single_support_swing_contact_weight = (
                args.forward_phase_single_support_swing_contact_weight
            )
        if args.forward_phase_single_support_stance_no_contact_weight is not None:
            config.reward_config.forward_phase_single_support_stance_no_contact_weight = (
                args.forward_phase_single_support_stance_no_contact_weight
            )
        if args.forward_swing_phase_advance_ticks is not None:
            config.reward_config.forward_swing_phase_advance_ticks = (
                args.forward_swing_phase_advance_ticks
            )
        if args.forward_swing_balance_grace_steps is not None:
            config.reward_config.forward_swing_balance_grace_steps = (
                args.forward_swing_balance_grace_steps
            )
        if args.forward_swing_advance_target_m is not None:
            config.reward_config.forward_swing_advance_target_m = (
                args.forward_swing_advance_target_m
            )
        swing_rate_joint_indices = _parse_int_list(
            args.forward_swing_target_rate_limit_joint_indices
        )
        if swing_rate_joint_indices is not None:
            config.reward_config.forward_swing_target_rate_limit_joint_indices = (
                swing_rate_joint_indices
            )
        swing_rate_limits = _parse_float_list(
            args.forward_swing_target_rate_limit_values
        )
        if swing_rate_limits is not None:
            config.reward_config.forward_swing_target_rate_limit_values = (
                swing_rate_limits
            )
        if args.push_recovery_tracking_window_steps is not None:
            config.reward_config.push_recovery_tracking_window_steps = (
                args.push_recovery_tracking_window_steps
            )
        push_recovery_joint_indices = _parse_int_list(
            args.push_recovery_tracking_joint_indices
        )
        if push_recovery_joint_indices is not None:
            config.reward_config.push_recovery_tracking_joint_indices = (
                push_recovery_joint_indices
            )
        reward_huber_overrides = {
            "action_rate_huber_delta": args.action_rate_huber_delta,
            "action_magnitude_huber_delta": args.action_magnitude_huber_delta,
            "target_rate_huber_delta": args.target_rate_huber_delta,
            "actuator_tracking_huber_delta": args.actuator_tracking_huber_delta,
            "push_recovery_actuator_tracking_huber_delta": (
                args.push_recovery_actuator_tracking_huber_delta
            ),
            "forward_shortfall_huber_delta": args.forward_shortfall_huber_delta,
            "forward_overshoot_huber_delta": args.forward_overshoot_huber_delta,
            "forward_wrong_direction_huber_delta": (
                args.forward_wrong_direction_huber_delta
            ),
            "forward_pitch_huber_delta": args.forward_pitch_huber_delta,
            "forward_pitch_rate_huber_delta": args.forward_pitch_rate_huber_delta,
            "forward_swing_clearance_huber_delta": (
                args.forward_swing_clearance_huber_delta
            ),
            "forward_phase_swing_lift_huber_delta": (
                args.forward_phase_swing_lift_huber_delta
            ),
            "forward_phase_single_support_swing_contact_weight": (
                args.forward_phase_single_support_swing_contact_weight
            ),
            "forward_phase_single_support_stance_no_contact_weight": (
                args.forward_phase_single_support_stance_no_contact_weight
            ),
            "forward_swing_phase_advance_ticks": (
                args.forward_swing_phase_advance_ticks
            ),
            "forward_swing_advance_huber_delta": (
                args.forward_swing_advance_huber_delta
            ),
            "forward_swing_target_rate_limit_huber_delta": (
                args.forward_swing_target_rate_limit_huber_delta
            ),
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

        if args.push_enable is not None:
            config.push_config.enable = args.push_enable
        if args.push_interval_min_s is not None or args.push_interval_max_s is not None:
            current_min, current_max = config.push_config.interval_range
            config.push_config.interval_range = [
                current_min if args.push_interval_min_s is None else args.push_interval_min_s,
                current_max if args.push_interval_max_s is None else args.push_interval_max_s,
            ]
        if args.push_magnitude_min is not None or args.push_magnitude_max is not None:
            current_min, current_max = config.push_config.magnitude_range
            config.push_config.magnitude_range = [
                current_min if args.push_magnitude_min is None else args.push_magnitude_min,
                current_max if args.push_magnitude_max is None else args.push_magnitude_max,
            ]

        if args.noise_level is not None:
            config.noise_config.level = args.noise_level
        noise_scale_overrides = {
            "hip_pos": args.noise_hip_pos,
            "knee_pos": args.noise_knee_pos,
            "ankle_pos": args.noise_ankle_pos,
            "joint_vel": args.noise_joint_vel,
            "gravity": args.noise_gravity,
            "gyro": args.noise_gyro,
            "accelerometer": args.noise_accelerometer,
        }
        for name, value in noise_scale_overrides.items():
            if value is not None:
                config.noise_config.scales[name] = value
        if hasattr(config, "reset_config"):
            reset_overrides = {
                "base_xy_jitter_m": args.reset_base_xy_jitter_m,
                "yaw_jitter_rad": args.reset_yaw_jitter_rad,
                "actuator_qpos_multiplier_min": (
                    args.reset_actuator_qpos_multiplier_min
                ),
                "actuator_qpos_multiplier_max": (
                    args.reset_actuator_qpos_multiplier_max
                ),
                "base_qvel_jitter": args.reset_base_qvel_jitter,
            }
            for name, value in reset_overrides.items():
                if value is not None:
                    config.reset_config[name] = value
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
        "--export_min_step",
        type=int,
        default=0,
        help=(
            "Skip checkpoint/ONNX export callbacks before this PPO step. "
            "Use a value above 0 for cloud training runs where step-0 export "
            "can destabilize the GPU process."
        ),
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
        "--enable_soft_prior",
        action="store_true",
        help="Enable default-off pitch-chain soft-prior auxiliary cost.",
    )
    parser.add_argument(
        "--soft_prior_config_json",
        type=str,
        default=None,
        help="Compact soft-prior config generated by build_soft_prior_fragment_config.py.",
    )
    parser.add_argument(
        "--soft_prior_scale",
        type=float,
        default=-0.05,
        help="Reward scale for soft_prior cost. Use a negative value to penalize.",
    )
    parser.add_argument("--soft_prior_huber_delta", type=float, default=0.05)
    parser.add_argument(
        "--soft_prior_phase_source",
        choices=["imitation_i", "step"],
        default="imitation_i",
    )
    parser.add_argument(
        "--enable_behavior_prior",
        action="store_true",
        help="Enable default-off state-conditioned frozen-MLP behavior prior.",
    )
    parser.add_argument(
        "--behavior_prior_mlp_npz",
        type=str,
        default=None,
        help="MLP NPZ with norm and w*/b* arrays for the behavior-prior teacher.",
    )
    parser.add_argument(
        "--behavior_prior_scale",
        type=float,
        default=-0.05,
        help="Reward scale for behavior_prior cost. Use a negative value to penalize.",
    )
    parser.add_argument("--behavior_prior_huber_delta", type=float, default=0.05)
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
        "--push_recovery_actuator_tracking_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales."
            "push_recovery_actuator_tracking. Use a negative value to penalize "
            "sent-vs-applied actuator mismatch only after push impulses."
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
        "--push_recovery_actuator_tracking_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for push-recovery actuator tracking "
            "cost. Default keeps the existing squared cost."
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
        "--forward_swing_clearance_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward swing-clearance cost. "
            "Default keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_phase_swing_lift_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for phase-commanded swing lift cost. "
            "Default keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_swing_advance_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward swing-advance cost. "
            "Default keeps the existing squared cost."
        ),
    )
    parser.add_argument(
        "--forward_swing_target_rate_limit_huber_delta",
        type=float,
        default=None,
        help=(
            "Optional pseudo-Huber delta for forward swing target-rate limit cost. "
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
        "--forward_single_support_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_single_support. "
            "Use a positive value to reward exactly one support foot under a "
            "forward command."
        ),
    )
    parser.add_argument(
        "--forward_double_support_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_double_support. "
            "Use a negative value to penalize double-support dwell under a "
            "forward command."
        ),
    )
    parser.add_argument(
        "--forward_contact_transition_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_contact_transition. "
            "Use a positive value to reward landing transitions that occur with "
            "body-frame forward progress."
        ),
    )
    parser.add_argument(
        "--forward_contact_transition_min_progress_ratio",
        type=float,
        default=None,
        help=(
            "Minimum command-normalized forward speed required for a contact "
            "transition reward."
        ),
    )
    parser.add_argument(
        "--forward_double_support_dwell_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_double_support_dwell. "
            "Use a negative value to penalize prolonged double support under a "
            "forward command."
        ),
    )
    parser.add_argument(
        "--forward_double_support_dwell_grace_steps",
        type=int,
        default=None,
        help="Number of forward-command double-support steps allowed before dwell cost grows.",
    )
    parser.add_argument(
        "--forward_swing_clearance_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_swing_clearance. "
            "Use a negative value to penalize touchdown after insufficient "
            "swing-foot lift during a forward command."
        ),
    )
    parser.add_argument(
        "--forward_swing_clearance_target_m",
        type=float,
        default=None,
        help=(
            "Target swing-foot peak lift over the last stance height, in meters, "
            "for forward_swing_clearance."
        ),
    )
    parser.add_argument(
        "--forward_phase_swing_lift_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_phase_swing_lift. "
            "Use a negative value to penalize low current foot lift during the "
            "phase-commanded swing window."
        ),
    )
    parser.add_argument(
        "--forward_phase_swing_lift_target_m",
        type=float,
        default=None,
        help=(
            "Target current swing-foot lift over stance height, in meters, for "
            "forward_phase_swing_lift."
        ),
    )
    parser.add_argument(
        "--forward_phase_single_support_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_phase_single_support. "
            "Use a negative value to penalize phase-commanded swing windows where "
            "the swing foot remains planted or the stance foot unloads."
        ),
    )
    parser.add_argument(
        "--forward_phase_single_support_swing_contact_weight",
        type=float,
        default=None,
        help="Cost weight for phase-swing foot contact in forward_phase_single_support.",
    )
    parser.add_argument(
        "--forward_phase_single_support_stance_no_contact_weight",
        type=float,
        default=None,
        help="Cost weight for stance-foot no-contact in forward_phase_single_support.",
    )
    parser.add_argument(
        "--forward_swing_phase_advance_ticks",
        type=int,
        default=None,
        help=(
            "Advance phase-primary swing-side reward masks by this many control "
            "ticks. Default 0 preserves the environment phase."
        ),
    )
    parser.add_argument(
        "--forward_swing_balance_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_swing_balance. "
            "Use a negative value to penalize one-sided swing usage during a "
            "forward command window."
        ),
    )
    parser.add_argument(
        "--forward_swing_balance_grace_steps",
        type=int,
        default=None,
        help="Number of accumulated swing steps before swing-balance cost is active.",
    )
    parser.add_argument(
        "--forward_swing_advance_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_swing_advance. "
            "Use a negative value to penalize touchdown after insufficient "
            "forward swing-foot advance during a forward command."
        ),
    )
    parser.add_argument(
        "--forward_swing_advance_target_m",
        type=float,
        default=None,
        help=(
            "Target swing-foot forward advance in the body frame, in meters, "
            "for forward_swing_advance."
        ),
    )
    parser.add_argument(
        "--forward_swing_target_rate_limit_scale",
        type=float,
        default=None,
        help=(
            "Optional override for reward_config.scales.forward_swing_target_rate_limit. "
            "Use a negative value to penalize phase-commanded swing target-rate excess "
            "against selected per-joint limits."
        ),
    )
    parser.add_argument(
        "--forward_swing_target_rate_limit_joint_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated actuator indices for phase-commanded swing target-rate "
            "limit cost, e.g. 11,12,13 for the right pitch chain."
        ),
    )
    parser.add_argument(
        "--forward_swing_target_rate_limit_values",
        type=str,
        default=None,
        help=(
            "Comma-separated rad/s limits matching --forward_swing_target_rate_limit_joint_indices."
        ),
    )
    parser.add_argument(
        "--push_recovery_tracking_window_steps",
        type=int,
        default=None,
        help=(
            "Number of control ticks after each push impulse to apply "
            "push-recovery tracking cost."
        ),
    )
    parser.add_argument(
        "--push_recovery_tracking_joint_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated actuator indices for push-recovery tracking cost. "
            "Default uses all actuators; e.g. 3 targets left_knee."
        ),
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
    parser.add_argument("--reset_base_xy_jitter_m", type=float, default=None)
    parser.add_argument("--reset_yaw_jitter_rad", type=float, default=None)
    parser.add_argument("--reset_actuator_qpos_multiplier_min", type=float, default=None)
    parser.add_argument("--reset_actuator_qpos_multiplier_max", type=float, default=None)
    parser.add_argument("--reset_base_qvel_jitter", type=float, default=None)
    parser.add_argument("--dr_friction_min", type=float, default=None)
    parser.add_argument("--dr_friction_max", type=float, default=None)
    parser.add_argument("--dr_frictionloss_scale_min", type=float, default=None)
    parser.add_argument("--dr_frictionloss_scale_max", type=float, default=None)
    parser.add_argument("--dr_armature_scale_min", type=float, default=None)
    parser.add_argument("--dr_armature_scale_max", type=float, default=None)
    parser.add_argument("--dr_com_jitter_m", type=float, default=None)
    parser.add_argument("--dr_mass_scale_min", type=float, default=None)
    parser.add_argument("--dr_mass_scale_max", type=float, default=None)
    parser.add_argument("--dr_torso_mass_delta_min", type=float, default=None)
    parser.add_argument("--dr_torso_mass_delta_max", type=float, default=None)
    parser.add_argument("--dr_qpos_jitter_rad", type=float, default=None)
    parser.add_argument("--dr_actuator_gain_scale_min", type=float, default=None)
    parser.add_argument("--dr_actuator_gain_scale_max", type=float, default=None)
    parser.add_argument(
        "--dr_leg_geometry_jitter_scale",
        type=float,
        default=None,
        help="Default-off multiplicative body_pos jitter for leg-link geometry.",
    )
    parser.add_argument(
        "--push_enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable training push perturbations.",
    )
    parser.add_argument("--push_interval_min_s", type=float, default=None)
    parser.add_argument("--push_interval_max_s", type=float, default=None)
    parser.add_argument("--push_magnitude_min", type=float, default=None)
    parser.add_argument("--push_magnitude_max", type=float, default=None)
    parser.add_argument("--noise_level", type=float, default=None)
    parser.add_argument("--noise_hip_pos", type=float, default=None)
    parser.add_argument("--noise_knee_pos", type=float, default=None)
    parser.add_argument("--noise_ankle_pos", type=float, default=None)
    parser.add_argument("--noise_joint_vel", type=float, default=None)
    parser.add_argument("--noise_gravity", type=float, default=None)
    parser.add_argument("--noise_gyro", type=float, default=None)
    parser.add_argument("--noise_accelerometer", type=float, default=None)
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
    parser.add_argument(
        "--ppo_desired_kl",
        type=float,
        default=None,
        help="Optional Brax PPO desired_kl override for adaptive KL LR schedule.",
    )
    parser.add_argument(
        "--ppo_learning_rate_schedule",
        choices=["NONE", "ADAPTIVE_KL"],
        default=None,
        help="Optional Brax PPO learning_rate_schedule override.",
    )
    parser.add_argument(
        "--ppo_learning_rate_schedule_min_lr",
        type=float,
        default=None,
        help="Optional minimum LR for the Brax adaptive KL schedule.",
    )
    parser.add_argument(
        "--ppo_learning_rate_schedule_max_lr",
        type=float,
        default=None,
        help="Optional maximum LR for the Brax adaptive KL schedule.",
    )
    parser.add_argument(
        "--restore_policy_kl_scale",
        type=float,
        default=None,
        help=(
            "Optional PPO loss coefficient for KL(current policy || restored "
            "checkpoint policy) on rollout observations. Requires "
            "--restore_checkpoint_path."
        ),
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
