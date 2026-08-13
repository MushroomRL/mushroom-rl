"""
Unified PPO training script for MuJoCo Warp / vanilla MuJoCo envs.

Two modes, one script:

  1. SCALED (default) — the real training run. Large num_envs, graph capture
     on, tuned fragment length. This is what replaces the old normal_ppo.py.

         python normal_ppo.py --envs hopper walker half_cheetah --seed 1

  2. PARITY — the receipt run for the PR. Small n_envs, benchmark hyperparams
     (lr 3e-4, n_features 32, batch 32, spf 2000), csv logging for the
     parity plot. Both backends selectable via --backend.

         python normal_ppo.py --parity --backend vanilla --env hopper --seed 1
         python normal_ppo.py --parity --backend warp    --env hopper --seed 1

Any individual hyperparam can be overridden on the CLI regardless of mode.

Known constraint: n_envs=1 is unsupported on the dev branch — mushroom's
TorchApproximator._parse_single_output squeezes batch-of-1 predictions
(torch_approximator.py:123), which flattens actions and crashes any
VectorizedEnvironment at n_envs=1. Both modes default to n_envs>=2.
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
from mushroom_rl.environments.mujoco_envs.hopper import Hopper
from mushroom_rl.environments.mujoco_envs.walker_2d import Walker2D
from mushroom_rl.environments.mujoco_envs.half_cheetah import HalfCheetah
from mushroom_rl.environments.mujoco_envs.ant import Ant
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
    n_envs workers. That is only sane at parity-scale (n_envs<=~16);
    the vanilla path is not intended for scaled training.
    """
    assert n_envs >= 2, "n_envs=1 hits the batch-of-1 squeeze bug in dev"
    if env_name not in ENV_TABLE:
        raise ValueError(f"unknown env: {env_name}")

    Vanilla, Warp = ENV_TABLE[env_name]

    if backend == "vanilla":
        if n_envs > 16:
            warnings.warn(
                f"vanilla backend at n_envs={n_envs} forks {n_envs} processes; "
                "this is only sensible for parity runs (n_envs<=16). "
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
# Presets
# ---------------------------------------------------------------------------


def parity_preset():
    """mushroom-rl-benchmark Hopper-v3/Walker2d-v3 PPO config exactly."""
    return dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        n_features=32,
        batch_size=32,
        n_epochs_policy=10,
        eps_ppo=0.2,
        lam=0.95,
        ent_coeff=0.0,
        std_0=1.0,
        n_steps=30000,
        n_steps_per_fit=2000,
        n_episodes_test=10,
        n_envs=2,
        use_graph_capture=False,
        use_wandb=False,
    )


def scaled_preset(num_envs):
    """
    Scaled training config. n_steps_per_fit is sized to give a longer
    on-policy fragment per env than the earlier 7.5 steps/env/fit (which
    was almost certainly the cause of the R plateau in earlier scaled
    runs). Default target: ~200 env-steps per env per fit.
    """
    steps_per_env_per_fit = 200
    n_steps_per_fit = num_envs * steps_per_env_per_fit
    # keep roughly 10 fits per epoch, matching the old normal_ppo cadence
    n_steps = n_steps_per_fit * 10
    return dict(
        actor_lr=1e-4,
        critic_lr=1e-3,
        n_features=64,
        batch_size=1024,
        n_epochs_policy=10,
        eps_ppo=0.2,
        lam=0.95,
        ent_coeff=0.01,
        std_0=1.0,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=5,
        n_envs=num_envs,
        use_graph_capture=True,
        use_wandb=True,
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment(
    env_name, backend, seed, n_epochs, cfg, mode, device, save_agent, results_dir
):
    TorchUtils.set_default_device(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mdp = make_env(env_name, backend, cfg["n_envs"], cfg["use_graph_capture"])
    if hasattr(mdp, "seed"):
        try:
            mdp.seed(seed)
        except Exception:
            pass

    run_name = f"{mode}_{env_name}_{backend}_seed{seed}"
    hyperparams = dict(
        cfg, env=env_name, backend=backend, seed=seed, n_epochs=n_epochs, mode=mode
    )

    logger_kwargs = dict(results_dir=results_dir, use_timestamp=True)
    if cfg["use_wandb"]:
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
        f"{mode.upper()}  env={env_name}  backend={backend}  "
        f"seed={seed}  n_envs={cfg['n_envs']}"
    )
    logger.info(
        f"lr={cfg['actor_lr']}/{cfg['critic_lr']} feat={cfg['n_features']} "
        f"batch={cfg['batch_size']} spf={cfg['n_steps_per_fit']} "
        f"n_steps={cfg['n_steps']} standardization=on "
        f"graph_capture={cfg['use_graph_capture']}"
    )

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": cfg["critic_lr"]}},
        loss=F.mse_loss,
        n_features=cfg["n_features"],
        batch_size=cfg["batch_size"],
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
        use_cuda=torch.cuda.is_available(),
    )

    alg_params = dict(
        actor_optimizer={
            "class": optim.Adam,
            "params": {"lr": cfg["actor_lr"], "eps": 1e-5},
        },
        n_epochs_policy=cfg["n_epochs_policy"],
        batch_size=cfg["batch_size"],
        eps_ppo=cfg["eps_ppo"],
        lam=cfg["lam"],
        ent_coeff=cfg["ent_coeff"],
        critic_params=critic_params,
    )

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=cfg["std_0"],
        n_features=cfg["n_features"],
        use_cuda=torch.cuda.is_available(),
    )

    agent = PPO(mdp.info, policy, **alg_params)
    agent.add_core_preprocessor(StandardizationPreprocessor(mdp.info))

    core = Core(agent, mdp)
    core.set_logger(logger)

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{run_name}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "J", "R", "entropy", "mean_ep_len"])

    best_J = float("-inf")

    def evaluate(epoch):
        nonlocal best_J
        dataset = core.evaluate(n_episodes=cfg["n_episodes_test"], render=False)
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
        if cfg["use_wandb"]:
            try:
                logger.log_evaluation(epoch, J=J, R=R, entropy=E, mean_ep_len=L)
            except Exception:
                pass
        writer.writerow([epoch, J, R, E, L])
        csv_file.flush()

        if save_agent and J > best_J:
            best_J = J
            try:
                logger.log_best_agent(agent, J)
            except Exception as e:
                logger.info(f"log_best_agent failed: {e}")

    evaluate(0)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=cfg["n_steps"], n_steps_per_fit=cfg["n_steps_per_fit"])
        evaluate(it + 1)

    csv_file.close()
    mdp.stop()
    logger.info(f"done. results -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def apply_overrides(cfg, args):
    """Any --lr / --n_features / ... flags override the preset."""
    overrides = {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "n_features": args.n_features,
        "batch_size": args.batch_size,
        "n_epochs_policy": args.n_epochs_policy,
        "eps_ppo": args.eps_ppo,
        "lam": args.lam,
        "ent_coeff": args.ent_coeff,
        "std_0": args.std_0,
        "n_steps": args.n_steps,
        "n_steps_per_fit": args.n_steps_per_fit,
        "n_episodes_test": args.n_episodes_test,
        "n_envs": args.n_envs,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.graph_capture is not None:
        cfg["use_graph_capture"] = args.graph_capture
    if args.wandb is not None:
        cfg["use_wandb"] = args.wandb
    return cfg


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Use the mushroom-rl-benchmark PPO preset (small n_envs, "
        "matched hyperparams) instead of the scaled training preset.",
    )
    parser.add_argument(
        "--num_envs_scaled",
        type=int,
        default=4000,
        help="n_envs used by the scaled preset. Ignored in --parity mode "
        "unless --n_envs is also passed.",
    )
    parser.add_argument("--save_agent", action="store_true")
    parser.add_argument("--results_dir", type=str, default="./logs")

    # per-hyperparam overrides
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs_policy", type=int, default=None)
    parser.add_argument("--eps_ppo", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--ent_coeff", type=float, default=None)
    parser.add_argument("--std_0", type=float, default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--n_steps_per_fit", type=int, default=None)
    parser.add_argument("--n_episodes_test", type=int, default=None)
    parser.add_argument(
        "--graph_capture",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=None,
        help="Override preset's graph capture default.",
    )
    parser.add_argument(
        "--wandb",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=None,
        help="Override preset's wandb default.",
    )

    args = parser.parse_args()
    envs = [args.single_env] if args.single_env else args.envs

    for env_name in envs:
        cfg = parity_preset() if args.parity else scaled_preset(args.num_envs_scaled)
        cfg = apply_overrides(cfg, args)
        mode = "parity" if args.parity else "scaled"
        experiment(
            env_name=env_name,
            backend=args.backend,
            seed=args.seed,
            n_epochs=args.n_epochs,
            cfg=cfg,
            mode=mode,
            device=args.device,
            save_agent=args.save_agent,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
