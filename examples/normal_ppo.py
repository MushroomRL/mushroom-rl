import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.mujoco_warp_envs import AntWarp
from mushroom_rl.policy import GaussianTorchPolicy

from mushroom_rl.utils import TorchUtils

TorchUtils.set_default_device("cuda:0")


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
    save_agent=False,
):

    torch.random.seed()

    mdp = env_class(num_envs=num_envs)

    actor_lr = 1e-4
    critic_lr = 5e-4
    n_features = 256
    batch_size = 512
    n_epochs_policy = 10
    eps_ppo = 0.2
    lam = 0.95
    std_0 = 1.0
    ent_coeff = 0.02
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
        std_0=std_0,
    )

    wandb_kwargs = Logger.default_wandb_kwargs(
        "mushroom_rl_wandb_example",
        config=hyperparams,
        name=PPO.__name__,
    )

    logger = Logger(
        PPO.__name__,
        results_dir="./logs",
        use_timestamp=True,
        wandb_kwargs=wandb_kwargs,
    )

    logger.strong_line()
    logger.info(f"Experiment Algorithm: {PPO.__name__} env: {env_class.__name__}")

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

        dataset = core.evaluate(
            n_episodes=n_episodes_test,
            render=False,
        )

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


if __name__ == "__main__":
    experiment(
        env_class=AntWarp,
        num_envs=4000,  # --> make full number, i.e 8000?
        n_epochs=25,  # --> half it
        n_steps=600000,  # --> double the n_steps
        n_steps_per_fit=64000,  # num_envs * 8
        n_episodes_test=5,
        save_agent=False,
    )
    # Problem 1: we have diff performance with diff horizon, and worse performance when increasing num_envs.
    # Problem 2: GPU utilization. Two reasons: the env could be too easy thus the memory is much more than what we need to do computation in the env, so scale won't help. Then the algorithm computation but it shouldn't be too much of an issue in the evaluation. Third reason: could be that Warp kernels are not efficient enough. I should have that step vectorized,
    # Reason: we have basically different horizons at every fit. If we use the same horizon, we should see that with more env we either learn faster or got better R. We're seeing the opposite now because we're using less data. Keep batch size consistent and same.
