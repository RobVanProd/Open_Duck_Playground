"""Phase/context-modulated PPO policy network for Open Duck experiments.

This keeps the standard Brax PPO distribution contract while using the same
phase-modulated policy structure as the offline BC parent:

  obs -> fixed obs norm -> trunk -> hidden
  obs[context_indices] -> fixed context norm -> context MLP -> gamma,beta
  hidden * (1 + scale * tanh(gamma)) + scale * tanh(beta) -> loc
  logits = concat(loc, scale_logits)

The fixed normalizers are stored inside the policy params and stop-gradiented
during apply. They are part of the restored artifact so checkpoint export can be
self-describing, but PPO does not update them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
import jax
import jax.numpy as jnp
import numpy as np


def _activation(name: str):
    if name == "swish":
        return linen.swish
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"unsupported phase-modulated activation: {name}")


def _dense_init(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
) -> dict[str, jax.Array]:
    kernel_key, _ = jax.random.split(key)
    scale = jnp.sqrt(jnp.asarray(2.0 / max(in_dim + out_dim, 1), dtype=jnp.float32))
    return {
        "kernel": jax.random.normal(kernel_key, (in_dim, out_dim), dtype=jnp.float32)
        * scale,
        "bias": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _dense(data: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
    return data @ params["kernel"] + params["bias"]


def make_policy_params_from_npz(npz_path: str, scale_logit: float = -2.0) -> dict[str, Any]:
    """Loads offline BC phase-modulated params into the PPO policy pytree."""
    bc = np.load(npz_path, allow_pickle=True)
    trunk_hidden_sizes = [int(v) for v in bc["trunk_hidden_sizes"].tolist()]
    context_hidden_sizes = [int(v) for v in bc["context_hidden_sizes"].tolist()]
    policy_params: dict[str, Any] = {
        "obs_mean": jnp.asarray(bc["obs_norm"][0], dtype=jnp.float32),
        "obs_std": jnp.asarray(bc["obs_norm"][1], dtype=jnp.float32),
        "context_mean": jnp.asarray(bc["context_norm"][0], dtype=jnp.float32),
        "context_std": jnp.asarray(bc["context_norm"][1], dtype=jnp.float32),
        "modulation_scale": jnp.asarray(float(bc["modulation_scale"][0]), dtype=jnp.float32),
        "scale_logits": jnp.full((14,), float(scale_logit), dtype=jnp.float32),
    }
    for index in range(len(trunk_hidden_sizes)):
        policy_params[f"trunk_{index}"] = {
            "kernel": jnp.asarray(bc[f"trunk_w{index}"], dtype=jnp.float32),
            "bias": jnp.asarray(bc[f"trunk_b{index}"], dtype=jnp.float32),
        }
    for index in range(len(context_hidden_sizes) + 1):
        policy_params[f"context_{index}"] = {
            "kernel": jnp.asarray(bc[f"context_w{index}"], dtype=jnp.float32),
            "bias": jnp.asarray(bc[f"context_b{index}"], dtype=jnp.float32),
        }
    policy_params["output"] = {
        "kernel": jnp.asarray(bc["output_w"], dtype=jnp.float32),
        "bias": jnp.asarray(bc["output_b"], dtype=jnp.float32),
    }
    return {"params": policy_params}


def make_phase_modulated_policy_network(
    param_size: int,
    observation_size,
    *,
    hidden_layer_sizes: Sequence[int] = (512, 256),
    context_hidden_layer_sizes: Sequence[int] = (64,),
    context_indices: Sequence[int] = (6, 99, 100),
    activation: str = "swish",
    obs_key: str = "state",
    init_scale_logit: float = -2.0,
    modulation_scale: float = 0.5,
    **_: Any,
) -> networks.FeedForwardNetwork:
    """Creates a Brax FeedForwardNetwork with phase/context modulation."""
    if param_size % 2 != 0:
        raise ValueError(f"expected even tanh-normal param_size, got {param_size}")
    action_size = param_size // 2
    if action_size != 14:
        raise ValueError(f"phase-modulated Open Duck policy expects 14 actions, got {action_size}")
    trunk_hidden = tuple(int(v) for v in hidden_layer_sizes)
    context_hidden = tuple(int(v) for v in context_hidden_layer_sizes)
    context_idx = tuple(int(v) for v in context_indices)
    act = _activation(activation)

    if isinstance(observation_size, Mapping):
        obs_size = int(observation_size[obs_key][0])
    else:
        obs_size = int(observation_size)

    def init(key: jax.Array) -> dict[str, Any]:
        keys = iter(jax.random.split(key, len(trunk_hidden) + len(context_hidden) + 6))
        dims = [obs_size, *trunk_hidden]
        params: dict[str, Any] = {
            "obs_mean": jnp.zeros((obs_size,), dtype=jnp.float32),
            "obs_std": jnp.ones((obs_size,), dtype=jnp.float32),
            "context_mean": jnp.zeros((len(context_idx),), dtype=jnp.float32),
            "context_std": jnp.ones((len(context_idx),), dtype=jnp.float32),
            "modulation_scale": jnp.asarray(float(modulation_scale), dtype=jnp.float32),
            "scale_logits": jnp.full((action_size,), float(init_scale_logit), dtype=jnp.float32),
        }
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            params[f"trunk_{index}"] = _dense_init(next(keys), in_dim, out_dim)
        context_dims = [len(context_idx), *context_hidden, trunk_hidden[-1] * 2]
        for index, (in_dim, out_dim) in enumerate(
            zip(context_dims[:-1], context_dims[1:], strict=True)
        ):
            params[f"context_{index}"] = _dense_init(next(keys), in_dim, out_dim)
        params["output"] = _dense_init(next(keys), trunk_hidden[-1], action_size)
        return {"params": params}

    def apply(processor_params, policy_params: Mapping[str, Any], obs):
        del processor_params
        if isinstance(obs, Mapping):
            obs = obs[obs_key]
        p = policy_params["params"]
        raw = obs
        obs_mean = jax.lax.stop_gradient(p["obs_mean"])
        obs_std = jax.lax.stop_gradient(p["obs_std"])
        context_mean = jax.lax.stop_gradient(p["context_mean"])
        context_std = jax.lax.stop_gradient(p["context_std"])
        z = (raw - obs_mean) / obs_std
        for index in range(len(trunk_hidden)):
            z = act(_dense(z, p[f"trunk_{index}"]))
        hidden = z

        context_raw = jnp.take(raw, jnp.asarray(context_idx, dtype=jnp.int32), axis=-1)
        c = (context_raw - context_mean) / context_std
        for index in range(len(context_hidden) + 1):
            c = _dense(c, p[f"context_{index}"])
            if index < len(context_hidden):
                c = act(c)
        gamma, beta = jnp.split(c, 2, axis=-1)
        scale = jax.lax.stop_gradient(p["modulation_scale"])
        modulated = hidden * (1.0 + scale * jnp.tanh(gamma)) + scale * jnp.tanh(beta)
        loc = _dense(modulated, p["output"])
        scale_logits = jnp.broadcast_to(p["scale_logits"], loc.shape)
        return jnp.concatenate([loc, scale_logits], axis=-1)

    return networks.FeedForwardNetwork(init=init, apply=apply)


def make_phase_modulated_ppo_networks(
    observation_size,
    action_size: int,
    preprocess_observations_fn=None,
    policy_hidden_layer_sizes: Sequence[int] = (512, 256),
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: str = "swish",
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    context_indices: Sequence[int] = (6, 99, 100),
    context_hidden_layer_sizes: Sequence[int] = (64,),
    init_noise_std: float = 1.0,
    init_scale_logit: float = -2.0,
    modulation_scale: float = 0.5,
    **_: Any,
) -> ppo_networks.PPONetworks:
    """Makes PPO networks with a phase-modulated actor and default critic."""
    del init_noise_std
    if distribution_type != "tanh_normal":
        raise ValueError("phase-modulated PPO currently supports tanh_normal only")
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_phase_modulated_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        context_hidden_layer_sizes=context_hidden_layer_sizes,
        context_indices=context_indices,
        activation=activation,
        obs_key=policy_obs_key,
        init_scale_logit=init_scale_logit,
        modulation_scale=modulation_scale,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn or types.identity_observation_preprocessor,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=_activation(activation),
        obs_key=value_obs_key,
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
