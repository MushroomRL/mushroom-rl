import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.mujoco_warp_envs import (
    AntWarp,
    HopperWarp,
    HalfCheetahWarp,
    Walker2DWarp,
)
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import TorchUtils

TorchUtils.set_default_device("cuda:0")


ENV_REGISTRY = {
    "ant": AntWarp,
    "hopper": HopperWarp,
    "half_cheetah": HalfCheetahWarp,
    "walker_2d": Walker2DWarp,
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

        if torch.cuda.is_available():
            self.to("cuda")

    def forward(self, state, **kwargs):
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        state = torch.squeeze(state, 1)
        x = F.relu(self._h1(state))
        x = F.relu(self._h2(x))
        return self._h3(x)


def experiment(
    env_class,
    num_envs,
    n_epochs,
    n_steps,
    n_steps_per_fit,
    n_episodes_test,
    use_graph_capture,
    seed,
    save_agent,
    wandb_project,
    wandb_run_name,
):
    # Seed everything deterministically at the start of this process
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mdp = env_class(
        num_envs=num_envs,
        use_graph_capture=use_graph_capture,
    )
    mdp.seed(seed)

    actor_lr = 1e-4
    critic_lr = 1e-3
    n_features = 256
    batch_size = 1024
    n_epochs_policy = 10
    eps_ppo = 0.2
    lam = 0.95
    std_0 = 0.3
    ent_coeff = 0

    hyperparams = dict(
        env=env_class.__name__,
        num_envs=num_envs,
        n_epochs=n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=n_episodes_test,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        n_features=n_features,
        n_epochs_policy=n_epochs_policy,
        eps_ppo=eps_ppo,
        ent_coeff=ent_coeff,
        lam=lam,
        seed=seed,
        use_graph_capture=use_graph_capture,
        std_0=std_0,
    )

    # Build wandb kwargs explicitly so project and run name are unambiguous.
    # Logger.default_wandb_kwargs is inconsistent across mushroom_rl versions;
    # constructing the dict directly is safer.
    wandb_kwargs = dict(
        project=wandb_project,
        name=wandb_run_name,
        config=hyperparams,
        reinit=True,  # allow multiple wandb.init in one process (defensive)
    )

    logger = Logger(
        PPO.__name__,
        results_dir=f"./logs/{env_class.__name__}/seed_{seed}",
        use_timestamp=True,
        wandb_kwargs=wandb_kwargs,
    )

    logger.strong_line()
    logger.info(
        f"Experiment: {PPO.__name__} | env: {env_class.__name__} | seed: {seed} "
        f"| wandb: {wandb_project}/{wandb_run_name}"
    )

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
        use_cuda=torch.cuda.is_available(),
    )

    alg_params = dict(
        actor_optimizer={
            "class": optim.Adam,
            "params": {"lr": actor_lr, "eps": 1e-5},
        },
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps_ppo,
        lam=lam,
        ent_coeff=ent_coeff,
        critic_params=critic_params,
    )

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=std_0,
        n_features=n_features,
        use_cuda=torch.cuda.is_available(),
    )

    agent = PPO(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)
    core.set_logger(logger)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()

    state = dataset.state
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state)

    try:
        E = agent.policy.entropy(state).item()
    except TypeError:
        E = agent.policy.entropy().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    try:
        core.evaluate(n_episodes=1, render=True, record=True)
        logger.log_video(0)
    except Exception as e:
        logger.info(f"Skipping initial video: {e}")

    for it in trange(n_epochs, leave=False):
        core.learn(
            n_steps=n_steps,
            n_steps_per_fit=n_steps_per_fit,
        )

        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()

        state = dataset.state
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state)

        try:
            E = agent.policy.entropy(state).item()
        except TypeError:
            E = agent.policy.entropy().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E)

        if save_agent:
            logger.log_best_agent(agent, J)

        if it + 1 == 20:
            try:
                core.evaluate(n_episodes=1, render=False, record=False)
                logger.log_video(it + 1)
            except Exception as e:
                logger.info(f"Skipping video at epoch {it+1}: {e}")

    try:
        core.evaluate(n_episodes=1, render=True, record=True)
        logger.log_video(n_epochs)
    except Exception as e:
        logger.info(f"Skipping final video: {e}")

    mdp.stop()

    # Explicitly finish wandb so state doesn't leak if this process is reused
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"wandb.finish failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="Which env to train.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=4000)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=300000)
    parser.add_argument("--n_steps_per_fit", type=int, default=30000)
    parser.add_argument("--n_episodes_test", type=int, default=5)
    parser.add_argument("--use_graph_capture", action="store_true", default=True)
    parser.add_argument(
        "--no_graph_capture", dest="use_graph_capture", action="store_false"
    )
    parser.add_argument("--save_agent", action="store_true", default=False)
    parser.add_argument(
        "--wandb_project_prefix",
        type=str,
        default="mushroom",
        help="wandb project = f'{prefix}_{env}'. One project per env, all seeds grouped inside.",
    )
    args = parser.parse_args()

    env_class = ENV_REGISTRY[args.env]

    # One project per env, seed is the run name.
    # Group all seeds inside the same project so you can plot them together.
    wandb_project = f"{args.wandb_project_prefix}_{args.env}"
    wandb_run_name = f"seed_{args.seed}"

    experiment(
        env_class=env_class,
        num_envs=args.num_envs,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_fit=args.n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        use_graph_capture=args.use_graph_capture,
        seed=args.seed,
        save_agent=args.save_agent,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
