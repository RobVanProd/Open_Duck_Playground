# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    reward_forward_progress,
    cost_forward_shortfall,
    cost_torques,
    cost_action_rate,
    cost_action_magnitude,
    cost_stand_still,
    reward_alive,
    pseudo_huber_cost,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        actuator_bridge=config_dict.create(
            enable=False,
            delay_min_ticks=3,
            delay_max_ticks=8,
            tau_min_s=0.06,
            tau_max_s=0.14,
            velocity_limit_min_rad_s=2.5,
            velocity_limit_max_rad_s=4.7,
            per_joint_variation=0.15,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=2.5,
                tracking_ang_vel=6.0,
                torques=-1.0e-3,
                action_rate=-0.5,  # was -1.5
                action_magnitude=0.0,
                stand_still=-0.2,  # was -1.0 TODO try to relax this a bit ?
                target_rate=0.0,
                actuator_tracking=0.0,
                forward_progress=0.0,
                forward_shortfall=0.0,
                command_progress=0.0,
                command_progress_shortfall=0.0,
                alive=20.0,
                imitation=1.0,
            ),
            tracking_sigma=0.01,  # was working at 0.01
            forward_progress_deadband=0.02,
            forward_shortfall_required_ratio=0.5,
            command_progress_required_ratio=0.6,
            command_progress_warmup_steps=50,
            action_rate_huber_delta=0.0,
            action_magnitude_huber_delta=0.0,
            target_rate_huber_delta=0.0,
            actuator_tracking_huber_delta=0.0,
            forward_shortfall_huber_delta=0.0,
            command_progress_shortfall_huber_delta=0.0,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],  # [-1.0, 1.0]
        command_resample_steps=500,
        zero_command_probability=0.1,
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,  # to make it easier
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # multiply actual joints with noise (excluding floating base and backlash)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        rng, bridge_rng = jax.random.split(rng)
        bridge_delay_ticks, bridge_tau_s, bridge_velocity_limit_rad_s = (
            self._sample_actuator_bridge_params(bridge_rng)
        )

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "actuator_bridge_target_history": jp.tile(
                self._default_actuator,
                self._config.actuator_bridge.delay_max_ticks + 1,
            ),
            "actuator_bridge_applied_targets": self._default_actuator,
            "actuator_bridge_delay_ticks": bridge_delay_ticks,
            "actuator_bridge_tau_s": bridge_tau_s,
            "actuator_bridge_velocity_limit_rad_s": bridge_velocity_limit_rad_s,
            "actuator_bridge_tracking_cost": jp.zeros(()),
            "target_velocity_cost": jp.zeros(()),
            "command_progress_distance": jp.zeros(()),
            "command_progress_steps": jp.zeros((), dtype=jp.int32),
            "command_progress_ratio": jp.zeros(()),
            "command_progress_shortfall_cost": jp.zeros(()),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        metrics["diagnostic/target_velocity_cost"] = jp.zeros(())
        metrics["diagnostic/actuator_bridge_tracking_cost"] = jp.zeros(())
        metrics["diagnostic/actuator_bridge_delay_ticks"] = jp.zeros(())
        metrics["diagnostic/actuator_bridge_tau_mean_s"] = jp.zeros(())
        metrics["diagnostic/actuator_bridge_velocity_limit_mean_rad_s"] = jp.zeros(())
        metrics["diagnostic/command_progress_ratio"] = jp.zeros(())
        metrics["diagnostic/command_progress_shortfall_cost"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)


    def _sample_actuator_bridge_params(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample per-episode actuator bridge parameters.

        The bridge is disabled by default. When enabled, it models the measured
        real target-to-joint delay with a target delay queue, first-order lag,
        and effective velocity limit. Shapes are fixed by delay_max_ticks so the
        environment remains JAX-compilable.
        """
        cfg = self._config.actuator_bridge
        if not cfg.enable:
            return (
                jp.array(0, dtype=jp.int32),
                jp.full((self._actuators,), self.dt),
                jp.full((self._actuators,), self._config.max_motor_velocity),
            )

        rng, delay_rng, tau_rng, vel_rng, tau_jitter_rng, vel_jitter_rng = (
            jax.random.split(rng, 6)
        )
        delay_ticks = jax.random.randint(
            delay_rng,
            (),
            minval=cfg.delay_min_ticks,
            maxval=cfg.delay_max_ticks + 1,
            dtype=jp.int32,
        )
        tau_base = jax.random.uniform(
            tau_rng, (), minval=cfg.tau_min_s, maxval=cfg.tau_max_s
        )
        vel_base = jax.random.uniform(
            vel_rng,
            (),
            minval=cfg.velocity_limit_min_rad_s,
            maxval=cfg.velocity_limit_max_rad_s,
        )
        tau_jitter = jax.random.uniform(
            tau_jitter_rng,
            (self._actuators,),
            minval=1.0 - cfg.per_joint_variation,
            maxval=1.0 + cfg.per_joint_variation,
        )
        vel_jitter = jax.random.uniform(
            vel_jitter_rng,
            (self._actuators,),
            minval=1.0 - cfg.per_joint_variation,
            maxval=1.0 + cfg.per_joint_variation,
        )
        tau_s = jp.clip(tau_base * tau_jitter, cfg.tau_min_s, cfg.tau_max_s)
        velocity_limit = jp.clip(
            vel_base * vel_jitter,
            cfg.velocity_limit_min_rad_s,
            cfg.velocity_limit_max_rad_s,
        )
        return delay_ticks, tau_s, velocity_limit

    def _apply_actuator_bridge(
        self, info: dict[str, Any], sent_motor_targets: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Return the lagged plant target while preserving sent-target history."""
        history = (
            jp.roll(info["actuator_bridge_target_history"], self._actuators)
            .at[: self._actuators]
            .set(sent_motor_targets)
        )
        delayed_target = history.reshape((-1, self._actuators))[
            info["actuator_bridge_delay_ticks"]
        ]
        previous_applied = info["actuator_bridge_applied_targets"]
        tau_s = jp.maximum(info["actuator_bridge_tau_s"], 1.0e-4)
        alpha = 1.0 - jp.exp(-self.dt / tau_s)
        lagged_target = previous_applied + alpha * (delayed_target - previous_applied)
        max_step = info["actuator_bridge_velocity_limit_rad_s"] * self.dt
        applied_target = jp.clip(
            lagged_target,
            previous_applied - max_step,
            previous_applied + max_step,
        )
        info["actuator_bridge_target_history"] = history
        info["actuator_bridge_applied_targets"] = applied_target
        return applied_target, info

    def _update_command_window_progress(
        self, info: dict[str, Any], data: mjx.Data
    ) -> None:
        """Track cumulative signed forward progress over the current command."""
        command_x = info["command"][0]
        local_vx = self.get_local_linvel(data)[0]
        signed_vx = local_vx * jp.sign(command_x)
        info["command_progress_distance"] += signed_vx * self.dt
        info["command_progress_steps"] += 1

        elapsed_s = jp.maximum(
            info["command_progress_steps"].astype(jp.float32) * self.dt,
            self.dt,
        )
        target_distance = jp.maximum(jp.abs(command_x) * elapsed_s, 1.0e-6)
        progress_ratio = info["command_progress_distance"] / target_distance

        needs_progress = (
            jp.abs(command_x) > self._config.reward_config.forward_progress_deadband
        )
        warm_enough = (
            info["command_progress_steps"]
            >= self._config.reward_config.command_progress_warmup_steps
        )
        required_distance = (
            target_distance * self._config.reward_config.command_progress_required_ratio
        )
        shortfall = jp.clip(
            required_distance - info["command_progress_distance"], 0.0, None
        )
        normalized_shortfall = shortfall / target_distance

        info["command_progress_ratio"] = jp.nan_to_num(
            jp.where(needs_progress, progress_ratio, 0.0)
        )
        info["command_progress_shortfall_cost"] = jp.nan_to_num(
            jp.where(
                needs_progress & warm_enough,
                pseudo_huber_cost(
                    normalized_shortfall,
                    self._config.reward_config.command_progress_shortfall_huber_delta,
                ),
                0.0,
            )
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = (
                state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )  # not critical, is already moduloed in get_reference_motion
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],
                state.info["command"][1],
                state.info["command"][2],
                state.info["imitation_i"],
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # Handle action delay
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[
            action_idx[0]
        ]  # action with delay

        # self.action_filter.push(action_w_delay)
        # action_w_delay = self.action_filter.get_filtered_action()

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )  # floating base x,y
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        ####

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]

            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets
                - self._config.max_motor_velocity * self.dt,  # control dt
                prev_motor_targets
                + self._config.max_motor_velocity * self.dt,  # control dt
            )

        # motor_targets.at[5:9].set(state.info["command"][3:])  # head joints
        sent_motor_targets = motor_targets
        if not USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
        target_velocity = (sent_motor_targets - prev_motor_targets) / self.dt
        applied_motor_targets = sent_motor_targets
        if self._config.actuator_bridge.enable:
            applied_motor_targets, _ = self._apply_actuator_bridge(
                state.info, sent_motor_targets
            )
        state.info["target_velocity_cost"] = jp.mean(
            pseudo_huber_cost(
                target_velocity, self._config.reward_config.target_rate_huber_delta
            )
        )
        state.info["actuator_bridge_tracking_cost"] = jp.mean(
            pseudo_huber_cost(
                sent_motor_targets - applied_motor_targets,
                self._config.reward_config.actuator_tracking_huber_delta,
            )
        )
        data = mjx_env.step(
            self.mjx_model, state.data, applied_motor_targets, self.n_substeps
        )

        self._update_command_window_progress(state.info, data)
        state.info["motor_targets"] = sent_motor_targets

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        # FIXME
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action  # was
        # state.info["last_act"] = motor_targets  # became
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        reset_command_window = done | (
            state.info["step"] > self._config.command_resample_steps
        )
        state.info["command"] = jp.where(
            reset_command_window,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            reset_command_window,
            0,
            state.info["step"],
        )
        state.info["command_progress_distance"] = jp.where(
            reset_command_window, 0.0, state.info["command_progress_distance"]
        )
        state.info["command_progress_steps"] = jp.where(
            reset_command_window, 0, state.info["command_progress_steps"]
        )
        state.info["command_progress_ratio"] = jp.where(
            reset_command_window, 0.0, state.info["command_progress_ratio"]
        )
        state.info["command_progress_shortfall_cost"] = jp.where(
            reset_command_window, 0.0, state.info["command_progress_shortfall_cost"]
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        state.metrics["diagnostic/target_velocity_cost"] = state.info[
            "target_velocity_cost"
        ]
        state.metrics["diagnostic/actuator_bridge_tracking_cost"] = state.info[
            "actuator_bridge_tracking_cost"
        ]
        state.metrics["diagnostic/actuator_bridge_delay_ticks"] = state.info[
            "actuator_bridge_delay_ticks"
        ].astype(reward.dtype)
        state.metrics["diagnostic/actuator_bridge_tau_mean_s"] = jp.mean(
            state.info["actuator_bridge_tau_s"]
        )
        state.metrics["diagnostic/actuator_bridge_velocity_limit_mean_rad_s"] = jp.mean(
            state.info["actuator_bridge_velocity_limit_rad_s"]
        )
        state.metrics["diagnostic/command_progress_ratio"] = state.info[
            "command_progress_ratio"
        ]
        state.metrics["diagnostic/command_progress_shortfall_cost"] = state.info[
            "command_progress_shortfall_cost"
        ]

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        # accelerometer[0] += 1.3 # TODO testing
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 3
                noisy_joint_angles - self._default_actuator,  # 10
                noisy_joint_vel * self._config.dof_vel_scale,  # 10
                info["last_act"],  # 10
                info["last_last_act"],  # 10
                info["last_last_last_act"],  # 10
                info["motor_targets"],  # 10
                contact,  # 2
                # info["current_reference_motion"],
                # info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 10
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        ret = {
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data),
                self._config.reward_config.tracking_sigma,
            ),
            "forward_progress": reward_forward_progress(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.forward_progress_deadband,
            ),
            "forward_shortfall": cost_forward_shortfall(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.forward_shortfall_required_ratio,
                self._config.reward_config.forward_progress_deadband,
                self._config.reward_config.forward_shortfall_huber_delta,
            ),
            "command_progress": info["command_progress_ratio"],
            "command_progress_shortfall": info["command_progress_shortfall_cost"],
            # "orientation": cost_orientation(self.get_gravity(data)),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(
                action,
                info["last_act"],
                self._config.reward_config.action_rate_huber_delta,
            ),
            "action_magnitude": cost_action_magnitude(
                action, self._config.reward_config.action_magnitude_huber_delta
            ),
            "target_rate": info["target_velocity_cost"],
            "actuator_tracking": info["actuator_bridge_tracking_cost"],
            "alive": reward_alive(),
            "imitation": reward_imitation(  # FIXME, this reward is so adhoc...
                self.get_floating_base_qpos(data.qpos),  # floating base qpos
                self.get_floating_base_qvel(data.qvel),  # floating base qvel
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                contact,
                info["current_reference_motion"],
                info["command"],
                USE_IMITATION_REWARD,
            ),
            "stand_still": cost_stand_still(
                # info["command"], data.qpos[7:], data.qvel[6:], self._default_pose
                info["command"],
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                ignore_head=False,
            ),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 8)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        neck_pitch = jax.random.uniform(
            rng5,
            minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
        )

        head_pitch = jax.random.uniform(
            rng6,
            minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
        )

        head_yaw = jax.random.uniform(
            rng7,
            minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
            maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
        )

        head_roll = jax.random.uniform(
            rng8,
            minval=self._config.head_roll_range[0] * self._config.head_range_factor,
            maxval=self._config.head_roll_range[1] * self._config.head_range_factor,
        )

        # Keep the historical 10% zero-command curriculum by default, but let
        # bridge experiments disable it without changing normal training.
        return jp.where(
            jax.random.bernoulli(rng4, p=self._config.zero_command_probability),
            jp.zeros(7),
            jp.hstack(
                [
                    lin_vel_x,
                    lin_vel_y,
                    ang_vel_yaw,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                    head_roll,
                ]
            ),
        )
