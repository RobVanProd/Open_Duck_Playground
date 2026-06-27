"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import (
    checkpoint as ppo_checkpoint,
    losses as ppo_losses,
    networks as ppo_networks,
    train as ppo,
)
import jax.numpy as jnp
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints
        export_min_step = getattr(self.args, "export_min_step", 0)
        if current_step < export_min_step:
            print(
                f"Skipping checkpoint/export at step {current_step}; "
                f"export_min_step={export_min_step}"
            )
            return

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,  # may not work
            output_path=onnx_export_path
        )

    def maybe_patch_restore_policy_kl_loss(self) -> None:
        """Optionally add a PPO loss term that anchors to restored policy logits."""
        scale = getattr(self.args, "restore_policy_kl_scale", None)
        if scale is None or scale <= 0:
            return
        if self.restore_checkpoint_path is None:
            raise ValueError(
                "--restore_policy_kl_scale requires --restore_checkpoint_path"
            )

        teacher_normalizer, teacher_policy_params, _ = ppo_checkpoint.load(
            self.restore_checkpoint_path
        )
        original_compute_ppo_loss = ppo_losses.compute_ppo_loss

        def compute_ppo_loss_with_restore_kl(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            *args,
            **kwargs,
        ):
            total_loss, metrics = original_compute_ppo_loss(
                params,
                normalizer_params,
                data,
                rng,
                ppo_network,
                *args,
                **kwargs,
            )
            data_t = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
            policy_apply = ppo_network.policy_network.apply
            dist = ppo_network.parametric_action_distribution
            current_logits = policy_apply(
                normalizer_params, params.policy, data_t.observation
            )
            teacher_logits = jax.lax.stop_gradient(
                policy_apply(
                    teacher_normalizer, teacher_policy_params, data_t.observation
                )
            )
            current_dist = dist.create_dist(current_logits)
            teacher_dist = dist.create_dist(teacher_logits)
            if hasattr(current_dist, "kl_divergence"):
                restore_policy_kl = jnp.mean(
                    current_dist.kl_divergence(teacher_dist)
                )
            else:
                restore_policy_kl = jnp.array(0.0)
            restore_policy_kl_loss = scale * restore_policy_kl
            total_loss = total_loss + restore_policy_kl_loss
            metrics = {
                **metrics,
                "restore_policy_kl": restore_policy_kl,
                "restore_policy_kl_loss": restore_policy_kl_loss,
            }
            return total_loss, metrics

        ppo_losses.compute_ppo_loss = compute_ppo_loss_with_restore_kl
        print(
            "Enabled restore-policy KL loss: "
            f"scale={scale} restore_checkpoint_path={self.restore_checkpoint_path}"
        )

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        ppo_cli_overrides = {
            "num_envs": getattr(self.args, "ppo_num_envs", None),
            "num_evals": getattr(self.args, "ppo_num_evals", None),
            "episode_length": getattr(self.args, "ppo_episode_length", None),
            "unroll_length": getattr(self.args, "ppo_unroll_length", None),
            "batch_size": getattr(self.args, "ppo_batch_size", None),
            "num_minibatches": getattr(self.args, "ppo_num_minibatches", None),
            "num_updates_per_batch": getattr(
                self.args, "ppo_num_updates_per_batch", None
            ),
            "learning_rate": getattr(self.args, "ppo_learning_rate", None),
            "entropy_cost": getattr(self.args, "ppo_entropy_cost", None),
            "clipping_epsilon": getattr(self.args, "ppo_clipping_epsilon", None),
            "max_grad_norm": getattr(self.args, "ppo_max_grad_norm", None),
            "desired_kl": getattr(self.args, "ppo_desired_kl", None),
            "learning_rate_schedule": getattr(
                self.args, "ppo_learning_rate_schedule", None
            ),
            "learning_rate_schedule_min_lr": getattr(
                self.args, "ppo_learning_rate_schedule_min_lr", None
            ),
            "learning_rate_schedule_max_lr": getattr(
                self.args, "ppo_learning_rate_schedule_max_lr", None
            ),
        }
        for key, value in ppo_cli_overrides.items():
            if value is not None:
                self.ppo_training_params[key] = value
        print(f"PPO params: {self.ppo_training_params}")
        self.maybe_patch_restore_policy_kl_loss()

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
