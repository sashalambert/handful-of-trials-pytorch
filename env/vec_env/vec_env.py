import numpy as np
import multiprocessing as mp
import gym
from .subproc_vec_env import SubprocVecEnv

class MPCVecEnv():

    def __init__(
            self,
            env_name,
            env_args,
            num_cpu=1,
    ):
        self.env_name = env_name
        self.env_args = env_args
        self.num_cpu = num_cpu
        if self.num_cpu is None:
            self.num_cpu = mp.cpu_count()

        self.env = None
        if self.num_cpu == 1:
            self.env = gym.make(env_name, **env_args)
        else:
            self.env = SubprocVecEnv(
                [self.make_env(
                    env_name,
                    env_args,
                    i,
                ) for i in range(self.num_cpu)
                ]
            )

    def make_env(
            self,
            env_name,
            env_args,
            rank,
            seed=0,
    ):
        """
        Utility function for multiprocessed env.
        :param env_name: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in
         subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_name, **env_args)
            env.seed(seed + rank)
            return env
        return _init

    def rollout(
            self,
            qpos,
            qvel,
            actions,
    ):
        """
        Rollout dynamics in open-loop fashion.
        """
        assert actions.shape[1] % self.num_cpu == 0,\
            "Number of samples must be divisible by number of cpus"

        # Split control batch
        action_blocks = np.array_split(actions, self.num_cpu, axis=1)
        self.env.rollout_async(
            qpos,
            qvel,
            action_blocks,
        )
        obs, rewards, dones = self.env.rollout_wait()

        return (
            actions,
            obs,
            rewards,
            dones,
        )
