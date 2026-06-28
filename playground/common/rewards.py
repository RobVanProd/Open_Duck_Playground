"""
Set of commonly used rewards
For examples on how to use some rewards, look at https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py
"""

import jax
import jax.numpy as jp


def pseudo_huber_cost(error: jax.Array, delta: float) -> jax.Array:
    """Quadratic near zero and linear for large residuals."""
    if delta <= 0.0:
        return jp.square(error)
    scaled = error / delta
    return jp.square(delta) * (jp.sqrt(1.0 + jp.square(scaled)) - 1.0)


# Tracking rewards.
def reward_tracking_lin_vel(
    commands: jax.Array,
    local_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    # lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    # return jp.nan_to_num(jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))
    y_tol = 0.1
    error_x = jp.square(commands[0] - local_vel[0])
    error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x + jp.square(error_y)
    return jp.nan_to_num(jp.exp(-lin_vel_error / tracking_sigma))


def reward_tracking_ang_vel(
    commands: jax.Array,
    ang_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.nan_to_num(jp.exp(-ang_vel_error / tracking_sigma))


def reward_forward_progress(
    commands: jax.Array,
    local_vel: jax.Array,
    deadband: float = 0.02,
) -> jax.Array:
    command_x = commands[0]
    needs_progress = jp.abs(command_x) > deadband
    target_speed = jp.maximum(jp.abs(command_x), 1.0e-6)
    signed_speed = local_vel[0] * jp.sign(command_x)
    progress_ratio = jp.clip(signed_speed / target_speed, 0.0, 1.0)
    return jp.nan_to_num(jp.where(needs_progress, progress_ratio, 0.0))


def cost_forward_shortfall(
    commands: jax.Array,
    local_vel: jax.Array,
    required_ratio: float = 0.5,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize standing still when a nonzero forward command is active."""
    command_x = commands[0]
    needs_progress = jp.abs(command_x) > deadband
    target_speed = jp.maximum(jp.abs(command_x), 1.0e-6)
    signed_speed = local_vel[0] * jp.sign(command_x)
    required_speed = target_speed * required_ratio
    shortfall = jp.clip(required_speed - signed_speed, 0.0, None)
    normalized_shortfall = shortfall / target_speed
    return jp.nan_to_num(
        jp.where(needs_progress, pseudo_huber_cost(normalized_shortfall, huber_delta), 0.0)
    )


def cost_forward_overshoot(
    commands: jax.Array,
    local_vel: jax.Array,
    allowed_ratio: float = 1.5,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize moving much faster than the active forward command."""
    command_x = commands[0]
    needs_progress = jp.abs(command_x) > deadband
    target_speed = jp.maximum(jp.abs(command_x), 1.0e-6)
    signed_speed = local_vel[0] * jp.sign(command_x)
    allowed_speed = target_speed * allowed_ratio
    overshoot = jp.clip(signed_speed - allowed_speed, 0.0, None)
    normalized_overshoot = overshoot / target_speed
    return jp.nan_to_num(
        jp.where(
            needs_progress,
            pseudo_huber_cost(normalized_overshoot, huber_delta),
            0.0,
        )
    )


def cost_forward_wrong_direction(
    commands: jax.Array,
    local_vel: jax.Array,
    allowed_reverse_ratio: float = 0.1,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize moving opposite the active forward command."""
    command_x = commands[0]
    needs_progress = jp.abs(command_x) > deadband
    target_speed = jp.maximum(jp.abs(command_x), 1.0e-6)
    signed_speed = local_vel[0] * jp.sign(command_x)
    allowed_reverse_speed = -target_speed * allowed_reverse_ratio
    wrong_direction = jp.clip(allowed_reverse_speed - signed_speed, 0.0, None)
    normalized_wrong_direction = wrong_direction / target_speed
    return jp.nan_to_num(
        jp.where(
            needs_progress,
            pseudo_huber_cost(normalized_wrong_direction, huber_delta),
            0.0,
        )
    )


# Base-related rewards.


def cost_lin_vel_z(global_linvel) -> jax.Array:
    return jp.nan_to_num(jp.square(global_linvel[2]))


def cost_ang_vel_xy(global_angvel) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))


def cost_orientation(torso_zaxis: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))


def cost_base_height(base_height: jax.Array, base_height_target: float) -> jax.Array:
    return jp.nan_to_num(jp.square(base_height - base_height_target))


def cost_forward_pitch(
    commands: jax.Array,
    gravity: jax.Array,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize pitch-like gravity tilt only while a forward command is active."""
    needs_progress = jp.abs(commands[0]) > deadband
    return jp.nan_to_num(
        jp.where(needs_progress, pseudo_huber_cost(gravity[0], huber_delta), 0.0)
    )


def cost_forward_pitch_rate(
    commands: jax.Array,
    gyro: jax.Array,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize pitch-rate-like gyro motion during forward-command rollouts."""
    needs_progress = jp.abs(commands[0]) > deadband
    return jp.nan_to_num(
        jp.where(needs_progress, pseudo_huber_cost(gyro[1], huber_delta), 0.0)
    )


def cost_forward_contact_support(
    commands: jax.Array,
    contact: jax.Array,
    deadband: float = 0.02,
    no_contact_weight: float = 1.0,
    asymmetry_weight: float = 0.0,
) -> jax.Array:
    """Penalize unsupported or optionally one-sided support under forward command."""
    needs_progress = jp.abs(commands[0]) > deadband
    contact_count = jp.sum(contact.astype(jp.float32))
    no_contact = contact_count < 0.5
    one_sided = jp.abs(contact[0].astype(jp.float32) - contact[1].astype(jp.float32))
    cost = (
        no_contact_weight * no_contact.astype(jp.float32)
        + asymmetry_weight * one_sided
    )
    return jp.nan_to_num(jp.where(needs_progress, cost, 0.0))


def reward_forward_single_support(
    commands: jax.Array,
    contact: jax.Array,
    deadband: float = 0.02,
) -> jax.Array:
    """Reward exactly one support foot while a forward command is active."""
    needs_progress = jp.abs(commands[0]) > deadband
    contact_count = jp.sum(contact.astype(jp.float32))
    single_support = jp.abs(contact_count - 1.0) < 0.5
    return jp.nan_to_num(jp.where(needs_progress, single_support.astype(jp.float32), 0.0))


def cost_forward_double_support(
    commands: jax.Array,
    contact: jax.Array,
    deadband: float = 0.02,
) -> jax.Array:
    """Penalize double-support dwell while a forward command is active."""
    needs_progress = jp.abs(commands[0]) > deadband
    contact_count = jp.sum(contact.astype(jp.float32))
    double_support = contact_count > 1.5
    return jp.nan_to_num(jp.where(needs_progress, double_support.astype(jp.float32), 0.0))


def reward_forward_contact_transition(
    commands: jax.Array,
    first_contact: jax.Array,
    local_vel: jax.Array,
    deadband: float = 0.02,
    min_progress_ratio: float = 0.25,
) -> jax.Array:
    """Reward a landing transition only when it coincides with forward progress."""
    command_x = commands[0]
    needs_progress = jp.abs(command_x) > deadband
    target_speed = jp.maximum(jp.abs(command_x), 1.0e-6)
    signed_speed = local_vel[0] * jp.sign(command_x)
    progress_ratio = signed_speed / target_speed
    has_forward_progress = progress_ratio >= min_progress_ratio
    any_first_contact = jp.any(first_contact).astype(jp.float32)
    return jp.nan_to_num(
        jp.where(
            needs_progress & has_forward_progress,
            any_first_contact * jp.clip(progress_ratio, 0.0, 1.0),
            0.0,
        )
    )


def cost_forward_double_support_dwell(
    commands: jax.Array,
    double_support_steps: jax.Array,
    grace_steps: int = 10,
    deadband: float = 0.02,
) -> jax.Array:
    """Penalize prolonged double support during a forward command."""
    needs_progress = jp.abs(commands[0]) > deadband
    grace = jp.maximum(jp.asarray(grace_steps, dtype=jp.float32), 1.0)
    excess = jp.clip(double_support_steps.astype(jp.float32) - grace, 0.0, None)
    return jp.nan_to_num(jp.where(needs_progress, excess / grace, 0.0))


def cost_forward_swing_clearance(
    commands: jax.Array,
    swing_peak_lift: jax.Array,
    first_contact: jax.Array,
    target_lift: float = 0.03,
    deadband: float = 0.02,
    huber_delta: float = 0.0,
) -> jax.Array:
    """Penalize low swing-foot peak lift on touchdown during a forward command."""
    needs_progress = jp.abs(commands[0]) > deadband
    shortfall = jp.clip(target_lift - swing_peak_lift, 0.0, None)
    cost = pseudo_huber_cost(shortfall, huber_delta)
    return jp.nan_to_num(jp.where(needs_progress, jp.sum(cost * first_contact), 0.0))


def cost_forward_swing_balance(
    commands: jax.Array,
    swing_steps: jax.Array,
    grace_steps: int = 20,
    deadband: float = 0.02,
) -> jax.Array:
    """Penalize one-sided swing usage during a forward command window."""
    needs_progress = jp.abs(commands[0]) > deadband
    swing = swing_steps.astype(jp.float32)
    total = jp.sum(swing)
    grace = jp.asarray(grace_steps, dtype=jp.float32)
    enough_samples = total > grace
    imbalance = jp.abs(swing[0] - swing[1]) / jp.maximum(total, 1.0)
    return jp.nan_to_num(jp.where(needs_progress & enough_samples, imbalance, 0.0))


def reward_base_y_swing(
    base_y_speed: jax.Array,
    freq: float,
    amplitude: float,
    t: float,
    tracking_sigma: float,
) -> jax.Array:
    target_y_speed = amplitude * jp.sin(2 * jp.pi * freq * t)
    y_speed_error = jp.square(target_y_speed - base_y_speed)
    return jp.nan_to_num(jp.exp(-y_speed_error / tracking_sigma))


# Energy related rewards.


def cost_torques(torques: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torques)))
    # return jp.nan_to_num(jp.sum(jp.abs(torques)))


def cost_energy(qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))


def cost_action_rate(
    act: jax.Array, last_act: jax.Array, huber_delta: float = 0.0
) -> jax.Array:
    c1 = jp.nan_to_num(jp.sum(pseudo_huber_cost(act - last_act, huber_delta)))
    return c1


def cost_action_magnitude(act: jax.Array, huber_delta: float = 0.0) -> jax.Array:
    return jp.nan_to_num(jp.sum(pseudo_huber_cost(act, huber_delta)))


# Other rewards.


def cost_joint_pos_limits(
    qpos: jax.Array, soft_lowers: float, soft_uppers: float
) -> jax.Array:
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.nan_to_num(jp.sum(out_of_limits))


def cost_stand_still(
    commands: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    default_pose: jax.Array,
    ignore_head: bool = False,
) -> jax.Array:
    # TODO no hard coded slices
    cmd_norm = jp.linalg.norm(commands[:3])
    if not ignore_head:
        pose_cost = jp.sum(jp.abs(qpos - default_pose))
        vel_cost = jp.sum(jp.abs(qvel))
    else:
        left_leg_pos = qpos[:5]
        right_leg_pos = qpos[9:]
        left_leg_vel = qvel[:5]
        right_leg_vel = qvel[9:]
        left_leg_default = default_pose[:5]
        right_leg_default = default_pose[9:]
        pose_cost = jp.sum(jp.abs(left_leg_pos - left_leg_default)) + jp.sum(
            jp.abs(right_leg_pos - right_leg_default)
        )
        vel_cost = jp.sum(jp.abs(left_leg_vel)) + jp.sum(jp.abs(right_leg_vel))

    return jp.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)


def cost_termination(done: jax.Array) -> jax.Array:
    return done


def reward_alive() -> jax.Array:
    return jp.array(1.0)


# Pose-related rewards.


def cost_head_pos(
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    cmd: jax.Array,
) -> jax.Array:
    move_cmd_norm = jp.linalg.norm(cmd[:3])
    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:9]
    # head_vel = joints_qvel[5:9]

    # target_head_qvel = jp.zeros_like(head_cmd)

    head_pos_error = jp.sum(jp.square(head_pos - head_cmd))

    # head_vel_error = jp.sum(jp.square(head_vel - target_head_qvel))

    return jp.nan_to_num(head_pos_error) * (move_cmd_norm > 0.01)
    # return jp.nan_to_num(head_pos_error + head_vel_error)


# FIXME
def cost_joint_deviation_hip(
    qpos: jax.Array, cmd: jax.Array, hip_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[hip_indices] - default_pose[hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return jp.nan_to_num(cost)


# FIXME
def cost_joint_deviation_knee(
    qpos: jax.Array, knee_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    return jp.nan_to_num(
        jp.sum(jp.abs(qpos[knee_indices] - default_pose[knee_indices]))
    )


# FIXME
def cost_pose(
    qpos: jax.Array, default_pose: jax.Array, weights: jax.Array
) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qpos - default_pose) * weights))


# Feet related rewards.


# FIXME
def cost_feet_slip(contact: jax.Array, global_linvel: jax.Array) -> jax.Array:
    body_vel = global_linvel[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return jp.nan_to_num(reward)


# FIXME
def cost_feet_clearance(
    feet_vel: jax.Array,
    foot_pos: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - max_foot_height)
    return jp.nan_to_num(jp.sum(delta * vel_norm))


# FIXME
def cost_feet_height(
    swing_peak: jax.Array,
    first_contact: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    error = swing_peak / max_foot_height - 1.0
    return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))


# FIXME
def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    threshold_min: float = 0.1,  # 0.2
    threshold_max: float = 0.5,
) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


# FIXME
def reward_feet_phase(
    foot_pos: jax.Array,
    rz: jax.Array,
) -> jax.Array:
    # Reward for tracking the desired foot height.
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return jp.nan_to_num(reward)
