#!/usr/bin/env python3
import gym
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from ruamel.yaml import YAML
import io
import tempfile

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
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1

import numpy as np

# class QuadrotorEnvActionWrapper:
#     """
#     Non-Gym wrapper for Flightmare's QuadrotorEnv_v1.
#     It intercepts the action before passing it to the underlying C++ environment.
#     """

#     def __init__(self, env, arm_length=0.1, k_drag=0.01, max_thrust=10.0, min_thrust=0.0):
#         self.env = env  # This is an instance of flightgym.QuadrotorEnv_v1
#         self.l = arm_length
#         self.k = k_drag
#         self.max_thrust = max_thrust
#         self.min_thrust = min_thrust

#         # Get dimensions from the C++ backend
#         self.obs_dim = self.env.getObsDim()
#         self.act_dim = self.env.getActDim()
#         self.extra_info_names = self.env.getExtraInfoNames()

#         # Preallocate buffers
#         self._obs = np.zeros(self.obs_dim, dtype=np.float32)
#         self._reward = np.zeros(1, dtype=np.float32)
#         self._done = np.zeros(1, dtype=np.bool)
#         self._extra = np.zeros((1, len(self.extra_info_names)), dtype=np.float32)

#         # Build allocation matrix and its inverse for torque→motor mapping
#         self.A = np.array([
#             [1, 1, 1, 1],
#             [0, self.l, 0, -self.l],
#             [-self.l, 0, self.l, 0],
#             [self.k, -self.k, self.k, -self.k]
#         ])
#         self.A_inv = np.linalg.inv(self.A)

#         # print("HELLOOOOOOOOOOOOOOOOOOOOOOOO", self.A_inv)

#     # --- Core modification ---
#     def mix_to_motors(self, torques_thrust):
#         """Convert [τx, τy, τz, T] → [f1, f2, f3, f4]."""
#         # tau_x, tau_y, tau_z, thrust = torques_thrust
#         # desired = np.array([thrust, tau_x, tau_y, tau_z])
#         # print(self.A_inv.shape)
#         # torques_thrust = np.array([1,3,2,5])
#         motor_forces = np.dot(self.A_inv,torques_thrust.T).T
#         print("MMMMMMMMMMMMMMMM", motor_forces)
#         return np.clip(motor_forces, self.min_thrust, self.max_thrust)

#     # --- API passthroughs ---
#     def reset(self, obs=None):
#         """
#         Compatible with FlightEnvVec. If an observation array is passed,
#         fill it in-place like the original QuadrotorEnv_v1.
#         """
#         if obs is not None:
#             # FlightEnvVec passes its own obs buffer
#             self.env.reset(obs)
#             return obs
#         else:
#             # Manual use (no obs passed)
#             self.env.reset(self._obs)
#             return self._obs.copy()

#     def step(self, action, obs=None, reward=None, done=None, extra=None):
#         """Modifies action before stepping the environment."""
#         action = np.asarray(action, dtype=np.float32)
#         print("ACTION", action.shape)
#         if action.shape[1] == 4:
#             # If policy outputs torques + thrust
#             print("helooooo")
#             mixed_action = self.mix_to_motors(action)
#         else:
#             # Already motor thrusts
#             mixed_action = action

#         self.env.step(mixed_action, self._obs, self._reward, self._done, self._extra)

#         info = {name: self._extra[0, i] for i, name in enumerate(self.extra_info_names)}
#         return self._obs.copy(), float(self._reward[0]), bool(self._done[0]), info

#     def seed(self, seed):
#         """Set environment seed."""
#         self.env.setSeed(int(seed))

#     def close(self):
#         """Closes the environment (if supported)."""
#         del self.env

#     # Optional: attribute passthrough for convenience
#     def __getattr__(self, name):
#         return getattr(self.env, name)

class QuadrotorEnvActionWrapper:
    """
    Non-Gym wrapper for Flightmare's QuadrotorEnv_v1.
    Intercepts actions [τx, τy, τz, T] and maps them to motor thrusts [f1..f4].
    Compatible with FlightEnvVec (C++-style interface) and direct Python-style calls.
    """

    def __init__(self, env, arm_length=0.1, k_drag=0.01, max_thrust=10.0, min_thrust=0.0):
        self.env = env  # instance of flightgym.QuadrotorEnv_v1
        self.l = arm_length
        self.k = k_drag
        self.max_thrust = float(max_thrust)
        self.min_thrust = float(min_thrust)

        # Support vectorized backend: get number of parallel envs
        # Some implementations expose getNumOfEnvs()
        self.num_envs = int(self.env.getNumOfEnvs()) if hasattr(self.env, "getNumOfEnvs") else 1

        # dims from backend
        self.obs_dim = int(self.env.getObsDim())
        self.act_dim = int(self.env.getActDim())
        self.extra_info_names = list(self.env.getExtraInfoNames())

        # Preallocate buffers with correct vectorized shapes
        self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        # reward and done shapes must match C++ signature: (m,1)
        self._reward = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._done = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self._extra = np.zeros((self.num_envs, max(1, len(self.extra_info_names))), dtype=np.float32)

        # Allocation matrix (maps motor thrusts -> [T, tau_x, tau_y, tau_z])
        # self.A = np.array([
        #     [1.0, 1.0, 1.0, 1.0],
        #     [0.0, self.l, 0.0, -self.l],
        #     [-self.l, 0.0, self.l, 0.0],
        #     [self.k, -self.k, self.k, -self.k]
        # ], dtype=np.float32)
        # self.A_inv = np.linalg.inv(self.A).astype(np.float32)

        L = 0.5
        K = 5

        self.A_inv = np.array([
            [1.0/4,  -1.0/(4*L),  1.0/(4*L),  -1.0/(4*K)],
            [1.0/4,   1.0/(4*L), -1.0/(4*L),  -1.0/(4*K)],
            [1.0/4,   1.0/(4*L),  1.0/(4*L),   1.0/(4*K)],
            [1.0/4,  -1.0/(4*L), -1.0/(4*L),   1.0/(4*K)]
])

    def mix_to_motors(self, torques_thrust):
        """
        Accepts:
          - single: shape (4,)  -> [tau_x, tau_y, tau_z, T]
          - batch:  shape (N,4) -> each row [tau_x, tau_y, tau_z, T]
        Returns:
          - single: shape (4,) -> [f1,f2,f3,f4]
          - batch:  shape (N,4)
        Note: reorders to [T, tau_x, tau_y, tau_z] before A_inv multiplication.
        """
        arr = np.asarray(torques_thrust, dtype=np.float32)

        if arr.ndim == 1:
            if arr.size != 4:
                raise ValueError("Expected action of length 4 ([τx,τy,τz,T]) or batched (N,4).")
            # reorder to [T, τx, τy, τz]
            desired = np.array([arr[3], arr[0], arr[1], arr[2]], dtype=np.float32)
            motor_forces = (self.A_inv @ desired).T  # shape (4,)
            motor_forces = np.clip(motor_forces, self.min_thrust, self.max_thrust)
            return motor_forces.astype(np.float32)

        elif arr.ndim == 2:
            if arr.shape[1] != 4:
                raise ValueError("Batched actions expected shape (N, 4).")
            # reorder columns to [T, τx, τy, τz] -> shape (N,4)
            desired_batch = arr[:, [3, 0, 1, 2]]  # (N,4)
            # compute A_inv @ desired_batch.T then transpose -> (N,4)
            motor_forces = (self.A_inv @ desired_batch.T).T
            motor_forces = np.clip(motor_forces, self.min_thrust, self.max_thrust)
            # print(f"MOTOR: {motor_forces}")
            return motor_forces.astype(np.float32)

        else:
            raise ValueError("Action array must be 1D (4,) or 2D (N,4).")

    def step(self, action, obs=None, reward=None, done=None, extra=None):
        """
        Two calling conventions supported:
        1) FlightEnvVec (C++ style) calls: step(action, obs_buf, reward_buf, done_buf, extra_buf)
           -> we must fill passed buffers in-place and return None.
        2) Direct Python call: step(action) -> returns (obs, reward, done, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # Normalize shape: ensure batched shape when FlightEnvVec uses multiple envs
        if action.ndim == 1:
            # single -> treat as (1, act_dim) when passing to C++ backend
            if action.size == 4:
                mixed = self.mix_to_motors(action)  # shape (4,)
            else:
                # if action already motor thrusts with different dimension
                mixed = action
            # Make it batched (1, act_dim)
            mixed_batched = mixed[np.newaxis, :].astype(np.float32)
        elif action.ndim == 2:
            # Batch of actions
            if action.shape[1] == 4:
                mixed_batched = self.mix_to_motors(action)  # returns (N,4)
            else:
                mixed_batched = action.astype(np.float32)
        else:
            raise ValueError("Action must be 1D or 2D array.")

        # If called by FlightEnvVec: fill the provided buffers in-place
        if obs is not None:
            # obs should be shape (m, obs_dim); reward (m,); done (m,) or (m,1)
            # Ensure correct shapes for C++ call: reward -> (m,1), done -> (m,1)
            # If passed buffers are 1D, try to reshape them to (m,1)
            # But normally FlightEnvVec already passes correct-shaped buffers.

            # Make sure intermediate arrays are C-contiguous and correct dtype as required by pybind11
            mixed_batched = np.ascontiguousarray(mixed_batched, dtype=np.float32)
            obs_buf = np.ascontiguousarray(obs, dtype=np.float32)
            reward_buf = np.ascontiguousarray(reward, dtype=np.float32)
            # done buffer expected boolean (m,1)
            done_buf = np.ascontiguousarray(done, dtype=np.bool_)
            extra_buf = np.ascontiguousarray(extra, dtype=np.float32)

            # Call underlying C++ step which fills buffers in-place
            self.env.step(mixed_batched, obs_buf, reward_buf, done_buf, extra_buf)
            # FlightEnvVec expects no return from wrapper.step in this mode
            return

        # Otherwise, direct Python-style call: we should use our internal buffers and return results
        # Ensure internal buffers have correct first dimension (num_envs)
        if mixed_batched.shape[0] != self.num_envs:
            # If mismatch, try to expand/shrink to match num_envs = 1 common case
            if self.num_envs == 1 and mixed_batched.shape[0] == 1:
                pass
            else:
                # Resize internal buffers to match mixed_batched batch size
                self.num_envs = mixed_batched.shape[0]
                self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
                self._reward = np.zeros((self.num_envs, 1), dtype=np.float32)
                self._done = np.zeros((self.num_envs, 1), dtype=np.bool_)
                self._extra = np.zeros((self.num_envs, max(1, len(self.extra_info_names))), dtype=np.float32)

        # Prepare C-contiguous arrays
        mixed_batched = np.ascontiguousarray(mixed_batched, dtype=np.float32)

        # Call the C++ step (fills our internal buffers)
        self.env.step(mixed_batched, self._obs, self._reward, self._done, self._extra)

        # Build python-style outputs:
        # If we have multiple envs, return full batched arrays; for single env, return flattened values
        if self.num_envs == 1:
            obs_out = self._obs[0].copy()
            reward_out = float(self._reward[0, 0])
            done_out = bool(self._done[0, 0])
            info = {'extra_info': {self.extra_info_names[i]: float(self._extra[0, i]) for i in range(len(self.extra_info_names))}}
            return obs_out, reward_out, done_out, info
        else:
            # for multi-env python-style call, return full arrays consistent with FlightEnvVec shape
            info = [{'extra_info': {self.extra_info_names[i]: float(self._extra[j, i]) for i in range(len(self.extra_info_names))}} for j in range(self.num_envs)]
            return self._obs.copy(), self._reward.copy().reshape(self.num_envs,), self._done.copy().reshape(self.num_envs,), info

    def reset(self, obs=None):
        """Compatible with FlightEnvVec (reset(obs)) and direct use (reset())."""
        if obs is not None:
            # fill provided obs buffer (must be shape (m, obs_dim))
            self.env.reset(obs)
            return obs
        else:
            # use internal buffer
            self.env.reset(self._obs)
            if self.num_envs == 1:
                return self._obs[0].copy()
            return self._obs.copy()

    def seed(self, seed):
        self.env.setSeed(int(seed))

    def close(self):
        del self.env

    def __getattr__(self, name):
        # forward anything we don't explicitly define to the inner env
        return getattr(self.env, name)
    

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
    parser.add_argument('-w', '--weight', type=str, default='./saved/2025-11-17-17-20-36_Iteration_542.zip',
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

    env = QuadrotorEnv_v1(tmp_file_path)

    env = QuadrotorEnvActionWrapper(env, arm_length=0.5)

    env = wrapper.FlightEnvVec(env)

    


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
            total_timesteps=int(25000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        print(args.weight)
        model = PPO2.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()

