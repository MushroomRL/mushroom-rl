import os
import numpy as np
import torch
from experiment_launcher.decorators import single_experiment
from experiment_launcher import run_experiment
import torch.optim as optim

from mushroom_rl.core import Logger, Core
from mushroom_rl.environments import Gymnasium

from mushroom_rl.algorithms.actor_critic import PPO_BPTT
from mushroom_rl.policy import RecurrentGaussianTorchPolicy

from tqdm import trange


def get_recurrent_network(rnn_type):
    if rnn_type == "vanilla":
        return torch.nn.RNN
    elif rnn_type == "gru":
        return torch.nn.GRU
    else:
        raise ValueError("Unknown RNN type %s." % rnn_type)
        

class PPOCriticBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1,
                 hidden_state_treatment="zero_initial", **kwargs):
        super().__init__()

        assert hidden_state_treatment in ["zero_initial", "use_policy_hidden_state"]

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._use_policy_hidden_states = True if hidden_state_treatment == "use_policy_hidden_state" else False

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._hq_1 = torch.nn.Linear(n_hidden_features+n_features, n_features)
        self._hq_2 = torch.nn.Linear(n_features, 1)
        self._act_func = torch.nn.ReLU()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq_1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self._hq_2.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, state, policy_state, lengths):
        # pre-rnn embedder
        input_rnn = self._act_func(self._h1_o(state))

        # --- forward rnn ---
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)
        if self._use_policy_hidden_states:
            # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
            # so we need to reshape and swap the first two axes.
            policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
            policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)
            out_rnn, _ = self._rnn(packed_seq, policy_state_reshaped)
        else:
            out_rnn, _ = self._rnn(packed_seq)   # use zero initial states

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_s = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_s, features_rnn], dim=1)
        q = self._hq_2(self._act_func(self._hq_1(input_last_layer)))

        return torch.squeeze(q)


class PPOActorBPTTNetwork(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_features, dim_env_state, rnn_type,
                 n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = get_recurrent_network(rnn_type)

        # embedder
        self._h1_o = torch.nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = torch.nn.Linear(dim_env_state, n_features)

        # rnn
        self._rnn = rnn(input_size=n_features,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        # nonlinearity=hidden_activation, # todo: this is turned off for now to allow for rnn and gru
                        batch_first=True)

        # post-rnn layer
        self._h3 = torch.nn.Linear(n_hidden_features+n_features, dim_action)
        self._act_func = torch.nn.ReLU()
        self._tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._h1_o.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)
        torch.nn.init.xavier_uniform_(self._h3.weight, gain=torch.nn.init.calculate_gain("relu")*0.05)

    def forward(self, state, policy_state, lengths):
        # pre-rnn embedder
        input_rnn = self._act_func(self._h1_o(state))

        # forward rnn
        # the inputs are padded. Based on that and the length, we created a packed sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                             batch_first=True)

        # hidden state has to have shape (N_layers, N_batch, DIM_hidden),
        # so we need to reshape and swap the first two axes.
        policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
        policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, policy_state_reshaped)

        # we only need the last entry in each sequence
        features_rnn, _ = torch.nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        # post-rnn embedder. Here we again only need the last state
        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        # last layer
        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)

        return a, torch.swapaxes(next_hidden, 0, 1)


def get_POMDP_params(pomdp_type):
    if pomdp_type == "no_velocities":
        return dict(obs_to_hide=("velocities",), random_force_com=False)
    elif pomdp_type == "no_positions":
        return dict(obs_to_hide=("positions",), random_force_com=False)
    elif pomdp_type == "windy":
        return dict(obs_to_hide=tuple(), random_force_com=True)


@single_experiment
def experiment(
        env: str = 'HalfCheetah-v4',
        horizon: int = 1000,
        gamma: float = 0.99,
        n_epochs: int = 300,
        n_steps_per_epoch: int = 50000,
        n_steps_per_fit: int = 2000,
        n_episode_eval: int = 10,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
        batch_size_actor: int = 32,
        batch_size_critic: int = 32,
        n_epochs_policy: int = 10,
        clip_eps_ppo: float = 0.05,
        gae_lambda: float = 0.95,
        seed: int = 0,  # This argument is mandatory
        results_dir: str = './logs',  # This argument is mandatory
        std_0: float = 0.5,
        rnn_type: str ="gru",
        n_hidden_features: int = 128,
        num_hidden_layers: int = 1,
        truncation_length: int = 5
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # prepare logging
    results_dir = os.path.join(results_dir, str(seed))
    logger = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed)

    # MDP
    mdp = Gymnasium(env, horizon=horizon, gamma=gamma)

    # create the policy
    dim_env_state = mdp.info.observation_space.shape[0]
    dim_action = mdp.info.action_space.shape[0]

    policy = RecurrentGaussianTorchPolicy(network=PPOActorBPTTNetwork,
                                          policy_state_shape=(n_hidden_features,),
                                          input_shape=(dim_env_state, ),
                                          output_shape=(dim_action,),
                                          n_features=128,
                                          rnn_type=rnn_type,
                                          n_hidden_features=n_hidden_features,
                                          num_hidden_layers=num_hidden_layers,
                                          dim_hidden_state=n_hidden_features,
                                          dim_env_state=dim_env_state,
                                          dim_action=dim_action,
                                          std_0=std_0)

    # setup critic
    input_shape_critic = (mdp.info.observation_space.shape[0]+2*n_hidden_features,)
    critic_params = dict(network=PPOCriticBPTTNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size_critic,
                         input_shape=input_shape_critic,
                         output_shape=(1,),
                         n_features=128,
                         n_hidden_features=n_hidden_features,
                         rnn_type=rnn_type,
                         num_hidden_layers=num_hidden_layers,
                         dim_env_state=mdp.info.observation_space.shape[0],
                         dim_hidden_state=n_hidden_features,
                         dim_action=dim_action
                         )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr': lr_actor,
                                                  'weight_decay': 0.0}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_actor,
                      dim_env_state=dim_env_state,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      truncation_length=truncation_length
                      )

    # Create the agent
    agent = PPO_BPTT(mdp_info=mdp.info, policy=policy, critic_params=critic_params, **alg_params)

    # Create Core
    core = Core(agent, mdp)

    # Evaluation
    dataset = core.evaluate(n_episodes=5)
    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    L = dataset.episodes_length.mean()
    logger.log_numpy(R=R, J=J, L=L)
    logger.epoch_info(0, R=R, J=J, L=L)

    for i in trange(1, n_epochs+1, 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit)

        # Evaluation
        dataset = core.evaluate(n_episodes=n_episode_eval)
        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        L = dataset.episodes_length.mean()
        logger.log_numpy(R=R, J=J, L=L)
        logger.epoch_info(i, R=R, J=J, L=L)


if __name__ == '__main__':
    run_experiment(experiment)
