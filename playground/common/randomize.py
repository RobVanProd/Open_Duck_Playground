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
"""Domain randomization for the Open Duck Mini V2 environment. (based on Berkeley Humanoid)"""

import jax
from mujoco import mjx
import jax.numpy as jp

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
LEG_GEOMETRY_BODY_IDS = jp.array([3, 4, 5, 6, 7, 13, 14, 15, 16, 17])


DEFAULT_CONFIG = {
    "friction_min": 0.5,
    "friction_max": 1.0,
    "frictionloss_scale_min": 0.9,
    "frictionloss_scale_max": 1.1,
    "armature_scale_min": 1.0,
    "armature_scale_max": 1.05,
    "com_jitter_m": 0.05,
    "mass_scale_min": 0.9,
    "mass_scale_max": 1.1,
    "torso_mass_delta_min": -0.1,
    "torso_mass_delta_max": 0.1,
    "qpos_jitter_rad": 0.03,
    "actuator_gain_scale_min": 0.9,
    "actuator_gain_scale_max": 1.1,
    "leg_geometry_jitter_scale": 0.0,
}


def make_domain_randomizer(config: dict | None = None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update({key: value for key, value in config.items() if value is not None})

    def randomizer(model: mjx.Model, rng: jax.Array):
        return domain_randomize(model, rng, cfg)

    return randomizer


def domain_randomize(model: mjx.Model, rng: jax.Array, config: dict | None = None):
    cfg = DEFAULT_CONFIG if config is None else config

    # _dof_addr=jp.array([6,8,10,12,14,16,18,20,22,24])
    # _joint_addr=jp.array([7,9,11,13,15,17,19,21,23,25])

    dof_id = jp.array(
        [idx for idx, fr in enumerate(model.dof_hasfrictionloss) if fr == True]
    )  # for backlash joint we disable frictionloss
    jnt_id = model.dof_jntid[dof_id]

    dof_addr = jp.array([jadd for jadd in model.jnt_dofadr if jadd in dof_id])
    joint_addr = model.jnt_qposadr[jnt_id]

    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(
                key, minval=cfg["friction_min"], maxval=cfg["friction_max"]
            )
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=cfg["frictionloss_scale_min"],
            maxval=cfg["frictionloss_scale_max"],
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[dof_addr] * jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=cfg["armature_scale_min"],
            maxval=cfg["armature_scale_max"],
        )
        dof_armature = model.dof_armature.at[dof_addr].set(armature)

        # Jitter center of mass positiion: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(
            key,
            (3,),
            minval=-cfg["com_jitter_m"],
            maxval=cfg["com_jitter_m"],
        )
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key,
            shape=(model.nbody,),
            minval=cfg["mass_scale_min"],
            maxval=cfg["mass_scale_max"],
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-0.2, 0.2).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key,
            minval=cfg["torso_mass_delta_min"],
            maxval=cfg["torso_mass_delta_max"],
        )
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[joint_addr].set(
            qpos0[joint_addr]
            + jax.random.uniform(
                key,
                shape=(model.nu,),
                minval=-cfg["qpos_jitter_rad"],
                maxval=cfg["qpos_jitter_rad"],
            )
        )

        # # Randomize KP
        rng, key = jax.random.split(rng)
        factor = jax.random.uniform(
            key,
            shape=(model.nu,),
            minval=cfg["actuator_gain_scale_min"],
            maxval=cfg["actuator_gain_scale_max"],
        )
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(current_kp * factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-current_kp * factor)

        rng, key = jax.random.split(rng)
        leg_scale = jax.random.uniform(
            key,
            shape=LEG_GEOMETRY_BODY_IDS.shape,
            minval=1.0 - cfg["leg_geometry_jitter_scale"],
            maxval=1.0 + cfg["leg_geometry_jitter_scale"],
        )
        body_pos = model.body_pos.at[LEG_GEOMETRY_BODY_IDS].set(
            model.body_pos[LEG_GEOMETRY_BODY_IDS] * leg_scale[:, None]
        )

        return (
            geom_friction,
            body_ipos,
            body_pos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        friction,
        body_ipos,
        body_pos,
        frictionloss,
        armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "body_pos": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "body_pos": body_pos,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes
