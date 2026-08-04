from multiprocessing import Pipe, Process, cpu_count

import numpy as np
import torch

from mushroom_rl.core.environment import Environment
from mushroom_rl.core.vectorized_env import VectorizedEnvironment


class _EnvWorker:
    """
    Body of a worker process: it owns one copy of the environment and serves the commands received on the
    pipe, sending back the result of each one, until it is told to close.

    """
    def __init__(self, remote, env_class, use_generator, args, kwargs):
        """
        Constructor.

        Args:
            remote (Connection): the worker end of the pipe connected to the main process;
            env_class (class): the environment class to be used;
            use_generator (bool): whether to use the generator to build the environment or not;
            args (tuple): the positional arguments to give to the constructor or to the generator of the class;
            kwargs (dict): the keyword arguments to give to the constructor or to the generator of the class;

        """
        self._remote = remote

        if use_generator:
            self._env = env_class.generate(*args, **kwargs)
        else:
            self._env = env_class(*args, **kwargs)

        self._handlers = dict(
            step=self._step,
            reset=self._reset,
            render=self._render,
            stop=self._stop,
            info=self._info,
            full_name=self._full_name,
            seed=self._seed
        )

        self._seed()

    @classmethod
    def run(cls, remote, env_class, use_generator, args, kwargs):
        """
        Entry point of a worker process: build the worker, then serve the commands sent by the main process
        until it asks to close. The environment is constructed here, and not in the main process, so that every
        copy lives in the process that steps it.

        Args:
            remote (Connection): the worker end of the pipe connected to the main process;
            env_class (class): the environment class to be used;
            use_generator (bool): whether to use the generator to build the environment or not;
            args (tuple): the positional arguments to give to the constructor or to the generator of the class;
            kwargs (dict): the keyword arguments to give to the constructor or to the generator of the class;

        """
        cls(remote, env_class, use_generator, args, kwargs)._serve()

    def _serve(self):
        """
        Serve the commands received on the pipe until the 'close' one arrives, replying to each of them.

        """
        try:
            while True:
                cmd, data = self._remote.recv()

                if cmd == 'close':
                    break

                handler = self._handlers.get(cmd)
                if handler is None:
                    raise NotImplementedError(f'Unknown command {cmd}')

                self._remote.send(handler(data))
        finally:
            self._remote.close()

    def _step(self, action):
        return self._env.step(action)

    def _reset(self, state):
        return self._env.reset(state)

    def _render(self, record):
        return self._env.render(record=record)

    def _stop(self, _):
        self._env.stop()

    def _info(self, _):
        return self._env.info

    def _full_name(self, _):
        return self._env.full_name()

    def _seed(self, seed=None):
        """
        Seed the random generators of the worker process, and the environment itself when it provides its own
        seeding. The worker is forked from the main process, so it starts with a copy of its generators:
        reseeding them is what keeps the copies of the environment from producing the very same trajectory.

        """
        if seed is None:
            np.random.seed(None)
            torch.seed()
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if type(self._env).seed is not Environment.seed:
            self._env.seed(seed)


class MultiprocessEnvironment(VectorizedEnvironment):
    """
    Basic interface to run in parallel multiple copies of the same environment.
    This class assumes that the environments are homogeneus, i.e. have the same type and MDP info.

    """
    def __init__(self, env_class, *args, n_envs=-1, use_generator=False, **kwargs):
        """
        Constructor.

        Args:
            env_class (class): The environment class to be used;
            *args: the positional arguments to give to the constructor or to the generator of the class;
            n_envs (int, -1): number of parallel copies of environment to construct;
            use_generator (bool, False): wheather to use the generator to build the environment or not;
            **kwargs: keyword arguments to set to the constructor or to the generator;

        """
        assert env_class is not None, "Environment class requires not installed module."
        assert n_envs > 1 or n_envs == -1

        if n_envs == -1:
            n_envs = cpu_count()

        self._remotes, self._work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self._processes = list()

        for work_remote in self._work_remotes:
            worker_process = Process(target=_EnvWorker.run,
                                     args=(work_remote, env_class, use_generator, args, kwargs))
            self._processes.append(worker_process)

        for p in self._processes:
            p.start()

        self._remotes[0].send(('info', None))
        mdp_info = self._remotes[0].recv()

        self._remotes[0].send(('full_name', None))
        self._env_name = self._remotes[0].recv()

        super().__init__(mdp_info, n_envs)

        self._state_shape = (n_envs,) + self.info.observation_space.shape
        self._reward_shape = (n_envs,)
        self._absorbing_shape = (n_envs,)
        self._states = np.empty(self._state_shape)

    def full_name(self):
        """
        Return a name identifying this environment and the one it runs copies of, joined by the same '.'
        separator used by the other environments wrapping a family of tasks. The wrapped name is the full
        name of a worker environment, so that the task it was built with is named as well.

        Returns:
            The name of the environment.

        """
        return f'{self.name()}.{self._env_name}'

    def reset_all(self, env_mask, state=None):
        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                state_i = state[i, :] if state is not None else None
                remote.send(('reset', state_i))

        episode_infos = list()
        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                state, episode_info = remote.recv()

                self._states[i] = state
                episode_infos.append(episode_info)
            else:
                episode_infos.append({})

        return self._states.copy(), episode_infos.copy()

    def step_all(self, env_mask, action):
        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                remote.send(('step', action[i, :]))

        rewards = np.empty(self._reward_shape)
        absorbings = np.zeros(self._absorbing_shape, dtype=bool)
        step_infos = list()

        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                state, reward, absorbing, step_info = remote.recv()

                self._states[i] = state
                rewards[i] = reward
                absorbings[i] = absorbing
                step_infos.append(step_info)
            else:
                step_infos.append({})

        return self._states.copy(), rewards.copy(), absorbings.copy(), step_infos.copy()

    def render_all(self, env_mask, record=False):
        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                remote.send(('render', record))

        frames = list()

        for i, remote in enumerate(self._remotes):
            if env_mask[i]:
                frame = remote.recv()
                frames.append(frame)

        return np.array(frames)

    def seed(self, seed):
        """
        Set the seed of every parallel copy of the environment. Each copy is given a different seed, derived
        from the given one, so that the copies do not generate the same trajectory.

        Args:
            seed (int, None): the value of the seed. If None, the random number generators of the copies are
                left untouched.

        """
        for i, remote in enumerate(self._remotes):
            remote.send(('seed', seed if seed is None else seed + i))

        for remote in self._remotes:
            remote.recv()

    def stop(self):
        for remote in self._remotes:
            remote.send(('stop', None))
            remote.recv()

    def close_all(self):
        """
        Terminate all the worker processes and join them. After this call the environment can no longer be used.

        """
        if getattr(self, '_closed', False):
            return
        self._closed = True

        if hasattr(self, '_remotes'):
            for remote in self._remotes:
                remote.send(('close', None))
        if hasattr(self, '_processes'):
            for p in self._processes:
                p.join()

    def __del__(self):
        self.close_all()

    @staticmethod
    def generate(env, *args, n_envs=-1, **kwargs):
        """
        Method to generate an array of multiple copies of the same environment, calling the generate method n_envs times

        Args:
            env (class): the environment to be constructed;
            *args: positional arguments to be passed to the constructor;
            n_envs (int, -1): number of environments to generate;
            **kwargs: keywords arguments to be passed to the constructor

        Returns:
            A list containing multiple copies of the environment.

        """
        use_generator = hasattr(env, 'generate')
        return MultiprocessEnvironment(env, *args, n_envs=n_envs, use_generator=use_generator, **kwargs)
