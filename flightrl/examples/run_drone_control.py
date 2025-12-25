#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from ruamel.yaml import YAML
import io
import tempfile
import tf2onnx

#
import os
import math
import argparse
import numpy as np
import tensorflow as tf

#
from stable_baselines import logger

#
from rpg_baselines.common.policies import MlpPolicy
# from stable_baselines3 import PPO
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
from tensorflow.python.framework import graph_util
# import stable_baselines3.common.utils as U
# import ruamel as YAML
#
from flightgym import QuadrotorEnv_v1


def export_onnx(model_path, name):

    # 1. Load your trained model
    # IMPORTANT: Ensure your environment matches the one you trained in
    # model_path = "my_sb1_model.zip"
    model = PPO2.load(model_path)

    # # 2. Access the internal TF session and graph
    # sess = model.sess
    # graph = sess.graph

    # # 3. Identify Input and Output Nodes
    # # Stable Baselines 1 usually names inputs 'input/Ob' (observations)
    # # and outputs depend on the action space (discrete vs continuous).
    # # We can print operations to find them if you aren't sure.

    # print("--- Potential Input/Output Nodes ---")
    # # Common input name in SB1
    # input_name = 'input/Ob:0' 

    # # Common output names:
    # # For PPO2/A2C (Stochastic/Deterministic policy output)
    # # Usually 'model/pi/add:0' or 'output/strided_slice:0' depending on implementation
    # # We will inspect the graph to be safe.
    # for op in graph.get_operations():
    #     if "input/Ob" in op.name:
    #         print(f"Input candidate: {op.name}")
    #     if "pi" in op.name and "add" in op.name: # Common for continuous actions
    #         print(f"Output candidate: {op.name}")
    #     if "strided_slice" in op.name: # Common for deterministic actions
    #         print(f"Output candidate: {op.name}")

    # # HARDCODED EXAMPLES (You may need to adjust these based on the prints above):
    # # Continuous Action Space (e.g., BipedalWalker):
    # output_node_names = ["model/pi/add"] 
    # # Discrete Action Space (e.g., CartPole):
    # # output_node_names = ["model/pi/probs"] or ["model/pi/log_softmax"]

    # # 4. Freeze the Graph
    # # This converts trained variables into constants directly in the graph
    # with graph.as_default():
    #     # We use tf2onnx to convert directly from the active session
    #     onnx_graph = tf2onnx.tfonnx.process_tf_graph(
    #         graph,
    #         input_names=[input_name],
    #         output_names=[n + ":0" for n in output_node_names],
    #         opset=11 # Opset 11 is widely supported by PyTorch/ORT
    #     )
        
    #     # 5. Save the ONNX model
    #     model_proto = onnx_graph.make_model("sb1_model")
    #     with open("sb1_exported.onnx", "wb") as f:
    #         f.write(model_proto.SerializeToString())

    # print("Successfully exported to sb1_exported.onnx")

    # model_path = "path_to_your_model.zip" # <--- UPDATE THIS
    # model = PPO2.load(model_path)
    sess = model.sess

    # 2. Define Input and Output Nodes based on your logs
    # Input: The observation placeholder
    input_node_names = ["input/Ob"] 
    input_tensor_names = ["input/Ob:0"]

    # Output: "model/pi/add" is usually the action mean in SB1 (Continuous Control)
    output_node_names = ["model/pi/add"] 
    output_tensor_names = ["model/pi/add:0"]

    print("Freezing the graph...")

    # 3. Freeze the Graph
    # This converts all "tf.float32_ref" variables into standard "tf.float32" constants
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names # giving it the list of output nodes tells it what to keep
    )

    # 4. Convert Frozen Graph to ONNX
    # We use from_graph_def instead of process_tf_graph to avoid the Reference error
    print("Converting to ONNX...")
    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=input_tensor_names,
        output_names=output_tensor_names,
        opset=11
    )

    # 5. Save
    output_path = name
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Success! Model saved to {output_path}")


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-19-11-44-04_Iteration_530.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-19-13-28-47_Iteration_787.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-17-08-17-12_Iteration_1796.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-17-14-14-30_Iteration_4392.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-18-09-12-38_Iteration_1398.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-18-13-51-47_Iteration_5322.zip',
    # parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-24-15-48-44_Iteration_18975.zip',
    parser.add_argument('-w', '--weight', type=str, default='./saved/2025-12-25-11-24-35_Iteration_3962.zip',
                        help='trained weight path')
    return parser


def main():
    # args = parser().parse_args()
    # cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
    #                        "/flightlib/configs/vec_env.yaml", 'r'))
    # if not args.train:
    #     cfg["env"]["num_envs"] = 1
    #     cfg["env"]["num_threads"] = 1

    # if args.render:
    #     cfg["env"]["render"] = "yes"
    # else:
    #     cfg["env"]["render"] = "no"

    # yaml = YAML() 
    # string_stream = io.StringIO() 
    # yaml.dump(cfg, string_stream) 
    # yaml_string = string_stream.getvalue()
    # print(yaml_string)
    # env = wrapper.FlightEnvVec(QuadrotorEnv_v1(yaml_string))

    args = parser().parse_args()
    yaml = YAML()

    # Load the base config
    cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    # Modify config according to args
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    cfg["env"]["render"] = "yes" if args.render else "no"

    # Save modified config to a temporary YAML file
    tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
    yaml.dump(cfg, tmp_file)
    tmp_file_path = tmp_file.name
    tmp_file.close()

    
    print(f"Using temporary config file: {tmp_file_path}")

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(tmp_file_path))

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # env.connectUnity()
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,  # check activation function  
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
            env=env,
            lam=0.95,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=250,
            ent_coef=0.00,
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
        )

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(250000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:

        # export_onnx(args.weight, "a_new_hope.onnx")

        model = PPO2.load(args.weight)

        # params = model.get_parameters()
        # np.save("sb1_params.npy", params)

        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
