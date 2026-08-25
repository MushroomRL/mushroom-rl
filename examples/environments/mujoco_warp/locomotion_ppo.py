"""
PPO training script for MuJoCo Warp locomotion envs.

"""

import argparse
import csv
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core, Logger, MultiprocessEnvironment
from mushroom_rl.environments.mujoco_envs.locomotion.hopper import Hopper
from mushroom_rl.environments.mujoco_envs.locomotion.walker_2d import Walker2D
from mushroom_rl.environments.mujoco_envs.locomotion.half_cheetah import HalfCheetah
from mushroom_rl.environments.mujoco_envs.locomotion.ant import Ant
from mushroom_rl.environments.mujoco_warp_envs import (
    AntWarp,
    HopperWarp,
    HalfCheetahWarp,
    Walker2DWarp,
)
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.utils import TorchUtils


ENV_TABLE = {
    "hopper": (Hopper, HopperWarp),
    "walker": (Walker2D, Walker2DWarp),
    "half_cheetah": (HalfCheetah, HalfCheetahWarp),
    "ant": (Ant, AntWarp),
}


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h3.weight, gain=nn.init.calculate_gain("linear") / 10
        )

    def forward(self, state, **kwargs):
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        state = torch.squeeze(state, 1) if state.dim() > 2 else state
        x = F.relu(self._h1(state))
        x = F.relu(self._h2(x))
        return self._h3(x)


def make_env(env_name, backend, n_envs, use_graph_capture):
    """
    Both backends run as VectorizedEnvironment at the same n_envs so the
    entire VectorizedCore collection path (masks, fit boundaries) is the
    same regardless of simulator.

    Vanilla at large n_envs uses MultiprocessEnvironment, which forks
    n_envs workers; only sensible at small scale (n_envs<=~16).
    """
    assert n_envs >= 2, "n_envs=1 hits the batch-of-1 squeeze bug in dev"
    if env_name not in ENV_TABLE:
        raise ValueError(f"unknown env: {env_name}")

    Vanilla, Warp = ENV_TABLE[env_name]

    if backend == "vanilla":
        if n_envs > 16:
            warnings.warn(
                f"vanilla backend at n_envs={n_envs} forks {n_envs} processes; "
                "this is only sensible at small scale (n_envs<=16). "
                "For scaled training use --backend warp.",
                stacklevel=2,
            )
        return MultiprocessEnvironment(Vanilla, n_envs=n_envs)

    return Warp(num_envs=n_envs, use_graph_capture=use_graph_capture)


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.float().mean().item()
    return float(np.mean(x))


def compute_entropy(agent, dataset):
    state = dataset.state
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(np.asarray(state))
    try:
        return agent.policy.entropy(state).item()
    except TypeError:
        return agent.policy.entropy().item()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment(env_name, args):
    TorchUtils.set_default_device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    mdp = make_env(env_name, args.backend, args.n_envs, args.graph_capture)
    if hasattr(mdp, "seed"):
        try:
            mdp.seed(args.seed)
        except Exception:
            pass

    run_name = f"{env_name}_{args.backend}_seed{args.seed}"
    hyperparams = dict(
        env=env_name,
        backend=args.backend,
        seed=args.seed,
        n_envs=args.n_envs,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_fit=args.n_steps_per_fit,
        n_epochs_policy=args.n_epochs_policy,
        n_episodes_test=args.n_episodes_test,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        n_features=args.n_features,
        batch_size=args.batch_size,
        eps_ppo=args.eps_ppo,
        lam=args.lam,
        ent_coeff=args.ent_coeff,
        std_0=args.std_0,
        graph_capture=args.graph_capture,
    )

    logger_kwargs = dict(results_dir=args.results_dir, use_timestamp=True)
    if args.wandb:
        try:
            logger_kwargs["wandb_kwargs"] = Logger.default_wandb_kwargs(
                f"mushroom_rl_{env_name}",
                config=hyperparams,
                name=run_name,
            )
        except Exception as e:
            print(f"wandb setup failed ({e}); continuing without wandb.")

    logger = Logger(run_name, **logger_kwargs)
    logger.strong_line()
    logger.info(
        f"env={env_name}  backend={args.backend}  "
        f"seed={args.seed}  n_envs={args.n_envs}"
    )
    logger.info(
        f"lr={args.actor_lr}/{args.critic_lr} feat={args.n_features} "
        f"batch={args.batch_size} spf={args.n_steps_per_fit} "
        f"n_steps={args.n_steps} ent_coeff={args.ent_coeff} "
        f"standardization=on graph_capture={args.graph_capture}"
    )

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": args.critic_lr}},
        loss=F.mse_loss,
        n_features=args.n_features,
        batch_size=args.batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
        use_cuda=torch.cuda.is_available(),
    )

    alg_params = dict(
        actor_optimizer={
            "class": optim.Adam,
            "params": {"lr": args.actor_lr, "eps": 1e-5},
        },
        n_epochs_policy=args.n_epochs_policy,
        batch_size=args.batch_size,
        eps_ppo=args.eps_ppo,
        lam=args.lam,
        ent_coeff=args.ent_coeff,
        critic_params=critic_params,
    )

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=args.std_0,
        n_features=args.n_features,
        use_cuda=torch.cuda.is_available(),
    )

    agent = PPO(mdp.info, policy, **alg_params)
    agent.add_core_preprocessor(StandardizationPreprocessor(mdp.info))

    core = Core(agent, mdp)
    core.set_logger(logger)

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, f"{run_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "J", "R", "entropy", "mean_ep_len"])

    best_J = float("-inf")

    def evaluate(epoch):
        nonlocal best_J
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=False)
        J = to_scalar(dataset.discounted_return)
        R = to_scalar(dataset.undiscounted_return)
        E = compute_entropy(agent, dataset)
        lengths = dataset.episodes_length
        L = to_scalar(
            lengths
            if isinstance(lengths, (np.ndarray, torch.Tensor))
            else np.asarray(lengths, dtype=np.float64)
        )
        logger.epoch_info(epoch, J=J, R=R, entropy=E, mean_ep_len=L)
        if args.wandb:
            try:
                logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L)
            except Exception:
                pass
        writer.writerow([epoch, J, R, E, L])
        csv_file.flush()

        if args.save_agent and J > best_J:
            best_J = J
            try:
                logger.log_best_agent(agent, J)
            except Exception as e:
                logger.info(f"log_best_agent failed: {e}")

    evaluate(0)

    for it in trange(args.n_epochs, leave=False):
        core.learn(n_steps=args.n_steps, n_steps_per_fit=args.n_steps_per_fit)
        evaluate(it + 1)

    csv_file.close()
    mdp.stop()
    logger.info(f"done. results -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def bool_flag(s):
    return s.lower() in ("1", "true", "yes")


def main():
    parser = argparse.ArgumentParser()

    # env / backend / seed / run control
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["hopper"],
        choices=list(ENV_TABLE.keys()),
        help="One or more envs to train sequentially.",
    )
    parser.add_argument(
        "--env",
        dest="single_env",
        default=None,
        choices=list(ENV_TABLE.keys()),
        help="Convenience alias for --envs with one entry.",
    )
    parser.add_argument("--backend", choices=["vanilla", "warp"], default="warp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="./logs")

    # training regime
    parser.add_argument("--n_envs", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=1_000_000)
    parser.add_argument("--n_steps_per_fit", type=int, default=100_000)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_epochs_policy", type=int, default=10)
    parser.add_argument("--n_episodes_test", type=int, default=10)

    # PPO / policy hyperparams
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--n_features", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eps_ppo", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--ent_coeff", type=float, default=0.001)
    parser.add_argument("--std_0", type=float, default=1.0)

    # toggles (default on; pass --flag false to disable)
    parser.add_argument("--graph_capture", type=bool_flag, default=True)
    parser.add_argument("--save_agent", type=bool_flag, default=True)
    parser.add_argument("--wandb", type=bool_flag, default=False)

    args = parser.parse_args()
    envs = [args.single_env] if args.single_env else args.envs

    for env_name in envs:
        experiment(env_name, args)


if __name__ == "__main__":
    main()
