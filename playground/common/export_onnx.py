import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import numpy as np

if os.environ.get("OPEN_DUCK_TF_EXPORT_ALLOW_GPU", "0") != "1":
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        print(f"TensorFlow GPU visibility already initialized: {exc}")


def _extract_policy_params(p):
    policy = getattr(p, 'policy', p)
    return policy.get('params') if isinstance(policy, dict) else None


def _config_get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _activation_node(nodes, input_name, output_name, activation, name):
    import onnx
    from onnx import helper

    del onnx
    if activation == "tanh":
        nodes.append(helper.make_node("Tanh", [input_name], [output_name], name=f"{name}_tanh"))
    elif activation == "swish":
        sigmoid = f"{name}_sigmoid"
        nodes.append(helper.make_node("Sigmoid", [input_name], [sigmoid], name=f"{name}_sigmoid"))
        nodes.append(helper.make_node("Mul", [input_name, sigmoid], [output_name], name=f"{name}_swish"))
    else:
        raise ValueError(f"unsupported phase-modulated ONNX activation: {activation}")


def _export_phase_modulated_onnx(
    policy_params,
    ppo_params,
    obs_size,
    output_path,
):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    network_factory = getattr(ppo_params, "network_factory", None)
    context_indices = _config_get(
        network_factory, "phase_modulated_context_indices", (6, 99, 100)
    )
    activation = _config_get(network_factory, "phase_modulated_activation", "swish")
    trunk_indices = sorted(
        int(key.split("_", 1)[1])
        for key in policy_params
        if key.startswith("trunk_") and key.split("_", 1)[1].isdigit()
    )
    context_layer_indices = sorted(
        int(key.split("_", 1)[1])
        for key in policy_params
        if key.startswith("context_") and key.split("_", 1)[1].isdigit()
    )

    def arr(name):
        return np.asarray(policy_params[name], dtype=np.float32)

    def layer_arr(layer_name, param_name):
        return np.asarray(policy_params[layer_name][param_name], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(arr("obs_mean"), name="obs_mean"),
        numpy_helper.from_array(arr("obs_std"), name="obs_std"),
        numpy_helper.from_array(np.asarray(context_indices, dtype=np.int64), name="context_indices"),
        numpy_helper.from_array(arr("context_mean"), name="context_mean"),
        numpy_helper.from_array(arr("context_std"), name="context_std"),
        numpy_helper.from_array(arr("modulation_scale").reshape(1), name="modulation_scale"),
        numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="one_initializer"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["obs_centered"], name="norm_obs_sub"),
        helper.make_node("Div", ["obs_centered", "obs_std"], ["trunk_in"], name="norm_obs_div"),
        helper.make_node("Gather", ["obs", "context_indices"], ["context_raw"], name="context_gather", axis=1),
        helper.make_node("Sub", ["context_raw", "context_mean"], ["context_centered"], name="norm_context_sub"),
        helper.make_node("Div", ["context_centered", "context_std"], ["context_in"], name="norm_context_div"),
    ]

    previous = "trunk_in"
    for index in trunk_indices:
        initializers.append(
            numpy_helper.from_array(layer_arr(f"trunk_{index}", "kernel"), name=f"trunk_w{index}")
        )
        initializers.append(
            numpy_helper.from_array(layer_arr(f"trunk_{index}", "bias"), name=f"trunk_b{index}")
        )
        gemm = f"trunk_gemm{index}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, f"trunk_w{index}", f"trunk_b{index}"],
                [gemm],
                name=f"trunk_gemm{index}",
            )
        )
        activated = f"trunk_act{index}"
        _activation_node(nodes, gemm, activated, activation, f"trunk_{index}")
        previous = activated
    hidden = previous

    previous = "context_in"
    for offset, index in enumerate(context_layer_indices):
        initializers.append(
            numpy_helper.from_array(layer_arr(f"context_{index}", "kernel"), name=f"context_w{index}")
        )
        initializers.append(
            numpy_helper.from_array(layer_arr(f"context_{index}", "bias"), name=f"context_b{index}")
        )
        gemm = f"context_gemm{index}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, f"context_w{index}", f"context_b{index}"],
                [gemm],
                name=f"context_gemm{index}",
            )
        )
        if offset < len(context_layer_indices) - 1:
            activated = f"context_act{index}"
            _activation_node(nodes, gemm, activated, activation, f"context_{index}")
            previous = activated
        else:
            previous = gemm

    hidden_dim = int(layer_arr("output", "kernel").shape[0])
    initializers.append(
        numpy_helper.from_array(np.asarray([hidden_dim, hidden_dim], dtype=np.int64), name="split_sizes")
    )
    nodes.extend(
        [
            helper.make_node("Split", [previous, "split_sizes"], ["gamma_raw", "beta_raw"], name="context_split", axis=1),
            helper.make_node("Tanh", ["gamma_raw"], ["gamma_tanh"], name="gamma_tanh"),
            helper.make_node("Tanh", ["beta_raw"], ["beta_tanh"], name="beta_tanh"),
            helper.make_node("Mul", ["gamma_tanh", "modulation_scale"], ["gamma_scaled"], name="gamma_scale"),
            helper.make_node("Mul", ["beta_tanh", "modulation_scale"], ["beta_scaled"], name="beta_scale"),
            helper.make_node("Add", ["gamma_scaled", "one_initializer"], ["gamma_plus_one"], name="gamma_plus_one"),
            helper.make_node("Mul", [hidden, "gamma_plus_one"], ["hidden_scaled"], name="hidden_apply_gamma"),
            helper.make_node("Add", ["hidden_scaled", "beta_scaled"], ["hidden_modulated"], name="hidden_apply_beta"),
        ]
    )
    initializers.append(
        numpy_helper.from_array(layer_arr("output", "kernel"), name="output_w")
    )
    initializers.append(
        numpy_helper.from_array(layer_arr("output", "bias"), name="output_b")
    )
    nodes.extend(
        [
            helper.make_node("Gemm", ["hidden_modulated", "output_w", "output_b"], ["loc"], name="output_gemm"),
            helper.make_node("Tanh", ["loc"], ["continuous_actions"], name="output_tanh"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "open_duck_phase_modulated_ppo_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, int(obs_size)])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, 14])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="open-duck-mini-rdkx5",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx"
):
    print(" === EXPORT ONNX === ")
    policy_params = _extract_policy_params(params[1])
    if isinstance(policy_params, dict) and "output" in policy_params and "obs_mean" in policy_params:
        print("Detected phase-modulated PPO policy; using phase-modulated ONNX exporter.")
        _export_phase_modulated_onnx(policy_params, ppo_params, obs_size, output_path)
        return

    # inference_fn = make_inference_fn(params, deterministic=True)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"hidden_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(
                        layers.LayerNormalization(name=f"layer_norm_{i}")
                    )
            if not self.activate_final and self.mlp_block.layers:
                if (
                    hasattr(self.mlp_block.layers[-1], "activation")
                    and self.mlp_block.layers[-1].activation is not None
                ):
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
        param_size,
        mean_std,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network

    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params["kernel"])
                bias = np.array(layer_params["bias"])
                print(
                    f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
                )
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    transfer_weights(policy_params, tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size), dtype=np.float32)]

    # Define the TensorFlow input signature
    spec = [
        tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
    ]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ["continuous_actions"]

    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=output_path
    )

    compat_output = os.environ.get("OPEN_DUCK_COMPAT_ONNX_OUTPUT")
    if compat_output:
        compat_path = Path(compat_output).expanduser()
        if not compat_path.is_absolute():
            compat_path = Path(output_path).resolve().parent / compat_path
        compat_path.parent.mkdir(parents=True, exist_ok=True)
        tf2onnx.convert.from_keras(
            tf_policy_network,
            input_signature=spec,
            opset=11,
            output_path=str(compat_path),
        )
    return
