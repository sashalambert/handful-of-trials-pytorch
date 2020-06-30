import multiprocessing

import gym
import numpy as np

from .base_vec_env import VecEnv, CloudpickleWrapper
from .tile_images import tile_images


def generate_env_rollout(
        env,
        qpos,
        qvel,
        Us,
):
    """
    :param env: mujoco environment
    :param qpos: current env qpos to initialize rollouts.
    :param qvel: current env qvel to initialize rollouts.
    :param Us: Control sequences, array of shape [steps, ctrl_samples,
    state_samples, control_dim]
    :return: Xs: state trajectories. array of shape [steps+1, ctrl_samples,
    state_samples, state_dim]
    """
    assert Us.ndim == 4
    (steps,
     num_ctrl_samples,
     num_state_samples,
     ctrl_dim) = Us.shape

    x0 = env._get_obs()
    state_dim = x0.shape[0]
    Xs = np.empty(
        (
            steps+1,
            num_ctrl_samples,
            num_state_samples,
            state_dim,
        ),
    )
    rewards = np.empty(
        steps,
        num_ctrl_samples,
        num_state_samples,
    )
    dones = np.empty_like(rewards)

    def reset_env(qpos, qvel):
        env.sim.set_state(qpos, qvel)

    for episode in range(num_ctrl_samples):
        for state_sample in range(num_state_samples):
            reset_env(qpos, qvel)
            Xs[0, ...] = x0
            for step in range(steps):
                (ob, reward, done, info) = env.step(
                    Us[step, episode, state_sample],
                )
                Xs[step+1, episode, state_sample, :] = ob
                rewards[step+1, episode, state_sample] = reward
                dones[step+1, episode, state_sample] = int(done)

    return Xs, rewards, dones

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'set_state':
                remote.send(env.set_state(data[0], data[1]))
            elif cmd == 'rollout':
                Xs = generate_env_rollout(env, data[0], data[1], data[2])
                remote.send(Xs)
            elif cmd == 'seed':
                np.random.seed(data)
                remote.send(env.seed(data))
            elif cmd == 'get_seed':
                remote.send((env.seed))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow
        sessions or other non thread-safe libraries are used in the parent (see issue #217).
        However, compared to 'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods,
        users must wrap the code in an ``if __name__ == "__main__":``
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'fork' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None, **kwargs):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            fork_available = 'fork' in multiprocessing.get_all_start_methods()
            start_method = 'fork' if fork_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        VecEnv.__init__(self, len(env_fns))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, np.stack(rews), np.stack(dones), infos

    def  rollout_async(self, start_state, action_seqs, cost_fn=None):
        """
        :param start_state: starting env state, array of size [state_env_dim]
        :param action_seqs: list of control actions, with elements of size [particles, state_dim]
        :param cost_fn: cost function to apply (Not currently used)
        """
        assert len(action_seqs) == len(self.remotes)

        for i,remote in enumerate(self.remotes):
            remote.send(('rollout', [start_state, action_seqs[i], cost_fn]))
        self.waiting = True

    def rollout_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting=False
        stacked_results = np.concatenate(results, axis=1)
        return stacked_results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def set_state_env(self, state_env):
        """
        :param state_env: Full state of environment.
        """
        for remote in self.remotes:
            remote.send(('set_state_env', state_env))
        for remote in self.remotes:
            remote.recv()

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def seed(self, seed_list):
        assert len(seed_list) == len(self.remotes), \
            "Each environment must be provided a seed"
        for i,remote in enumerate(self.remotes):
            remote.send(('seed', seed_list[i]))
        _ = [remote.recv() for remote in self.remotes]

