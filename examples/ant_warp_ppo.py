import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.algorithms.actor_critic import RudinPPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.mujoco_warp_envs import AntWarp
from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        if isinstance(n_features, int):
            n_features = [n_features]

        sizes = [n_input] + list(n_features) + [n_output]
        self._layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])

        for i, layer in enumerate(self._layers):
            is_last = (i == len(self._layers) - 1)
            gain = nn.init.calculate_gain("linear" if is_last else "relu") / 10
            nn.init.xavier_uniform_(layer.weight, gain=gain)

        if torch.cuda.is_available():
            self.to("cuda")

    def forward(self, state, **kwargs):
        device = next(self.parameters()).device
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        x = torch.squeeze(x, 1)
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i < len(self._layers) - 1:
                x = F.relu(x)
        return x


def experiment(env_class, num_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test):
    rollout_len = 24
    n_features = [512, 256, 128]
    batch_size = int((num_envs * rollout_len) / 32)
    actor_lr = 1e-3
    critic_lr = 1e-3
    n_epochs_policy = 5
    eps_ppo = 0.2
    lam = 0.95
    std_0 = 1.0
    ent_coeff = 0.0

    hyperparams = dict(
        alg="RudinPPO",
        env=env_class.__name__,
        num_envs=num_envs,
        rollout_len=rollout_len,
        n_epochs=n_epochs,
        n_steps=n_steps,
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_test=n_episodes_test,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        n_features=str(n_features),
        batch_size=batch_size,
        n_epochs_policy=n_epochs_policy,
        eps_ppo=eps_ppo,
        lam=lam,
        std_0=std_0,
        ent_coeff=ent_coeff,
    )

    try:
        wandb_kwargs = Logger.default_wandb_kwargs(
            "mushroom_warp_ppo",
            config=hyperparams,
            name=f"{RudinPPO.__name__}_{env_class.__name__}",
        )
        logger = Logger(
            RudinPPO.__name__ + f"_{env_class.__name__}",
            results_dir="./logs/",
            log_console=True,
            use_timestamp=True,
            wandb_kwargs=wandb_kwargs,
        )
        _pure_wandb = False
    except (AttributeError, TypeError):
        wandb.init(
            project="mushroom_warp_ppo",
            config=hyperparams,
            name=f"{RudinPPO.__name__}_{env_class.__name__}",
        )
        logger = Logger(
            RudinPPO.__name__ + f"_{env_class.__name__}",
            results_dir="./logs/",
            log_console=True,
            use_timestamp=True,
        )
        _pure_wandb = True

    logger.strong_line()
    logger.info(f"Experiment Algorithm: {RudinPPO.__name__}  env: {env_class.__name__}")

    mdp = env_class(num_envs=num_envs)

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        use_cuda=torch.cuda.is_available(),
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    alg_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
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

    agent = RudinPPO(mdp.info, policy, **alg_params)
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    E = agent.policy.entropy().item()
    logger.epoch_info(0, J=J, R=R, entropy=E)
    if _pure_wandb:
        wandb.log({"eval/J": J, "eval/R": R, "eval/entropy": E, "epoch": 0})
    del dataset

    best_R = -float("inf")
    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        E = agent.policy.entropy().item()
        logger.epoch_info(it + 1, J=J, R=R, entropy=E)
        if _pure_wandb:
            wandb.log({"eval/J": J, "eval/R": R, "eval/entropy": E, "epoch": it + 1})

        try:
            if R > best_R:
                best_R = R
                logger.log_best_agent(agent, R)
            logger.log_agent(agent)
        except AttributeError:
            agent.save(f"./logs/agent_epoch_{it+1:03d}.msh")
            if R > best_R:
                best_R = R
                agent.save("./logs/agent_best.msh")

        del dataset

    if _pure_wandb:
        wandb.finish()
    mdp.stop()


if __name__ == "__main__":
    num_envs = 4096
    rollout_len = 24

    envs = [AntWarp]
    for env in envs:
        experiment(
            env_class=env,
            num_envs=num_envs,
            n_epochs=30,
            n_steps=num_envs * rollout_len * 50 * 2,
            n_steps_per_fit=num_envs * rollout_len,
            n_episodes_test=256,
        )
