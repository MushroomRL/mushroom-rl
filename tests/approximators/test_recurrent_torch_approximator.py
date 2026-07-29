import pytest
import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.approximators.parametric import RecurrentTorchApproximator, RecurrentTorchEnsemble
from mushroom_rl.approximators.parametric.networks import RecurrentActorNetwork, RecurrentCriticNetwork


def _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden, action_history_shape=None):
    return RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=[(dim_action,), (n_hidden,)],
        network=RecurrentActorNetwork, policy_state_shape=(n_hidden,),
        n_features=8, rnn_type='gru', n_hidden_features=n_hidden, num_hidden_layers=1,
        action_history_shape=action_history_shape
    )


def test_recurrent_approximator_single_sample_padding():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden = 4, 2, 6
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden)

    action, next_state = approximator.predict(torch.randn(dim_env), torch.zeros(n_hidden))
    assert action.shape == (dim_action,)
    assert next_state.shape == (1, n_hidden)


def test_recurrent_approximator_vectorized_padding():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden, n_envs = 4, 2, 6, 3
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden)

    action, next_state = approximator.predict(torch.randn(n_envs, dim_env), torch.zeros(n_envs, n_hidden))
    assert action.shape == (n_envs, dim_action)
    assert next_state.shape == (n_envs, 1, n_hidden)


def test_recurrent_approximator_training_passthrough():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden, batch, seq = 4, 2, 6, 5, 4
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden)

    lengths = torch.tensor([4, 2, 3, 4, 1])
    action, next_state = approximator.predict(torch.randn(batch, seq, dim_env), torch.zeros(batch, n_hidden),
                                              lengths=lengths)
    assert action.shape == (batch, dim_action)
    assert next_state.shape == (batch, 1, n_hidden)


def test_recurrent_approximator_lengths_default_matches_explicit():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden = 4, 2, 6
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden)

    state, policy_state = torch.randn(dim_env), torch.zeros(n_hidden)
    default_action, _ = approximator.predict(state, policy_state)
    explicit_action, _ = approximator.predict(state, policy_state, lengths=torch.ones(1, dtype=torch.long))
    assert torch.allclose(default_action, explicit_action)


def test_recurrent_approximator_action_history_padding():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden = 4, 2, 6
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden,
                                                      action_history_shape=(dim_action,))

    action, next_state = approximator.predict(torch.randn(dim_env), torch.zeros(n_hidden),
                                              action_history=torch.randn(dim_action))
    assert action.shape == (dim_action,)
    assert next_state.shape == (1, n_hidden)


def test_recurrent_approximator_missing_action_history_raises():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden = 4, 2, 6
    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden,
                                                      action_history_shape=(dim_action,))

    with pytest.raises(KeyError):
        approximator.predict(torch.randn(dim_env), torch.zeros(n_hidden))


def test_recurrent_ensemble():
    torch.manual_seed(1)
    dim_env, n_hidden, batch, seq = 4, 6, 8, 5

    approximator = RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=(1,), network=RecurrentCriticNetwork,
        policy_state_shape=(n_hidden,), n_models=3, prediction='all', n_features=8, rnn_type='gru',
        n_hidden_features=n_hidden, num_hidden_layers=1,
        optimizer={'class': optim.Adam, 'params': {}}, loss=F.mse_loss, batch_size=0, quiet=True
    )
    assert isinstance(approximator, RecurrentTorchEnsemble)
    assert len(approximator) == 3

    state = torch.randn(batch, seq, dim_env)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 1, 5])
    target = torch.randn(batch)

    weights_before = approximator.get_weights().clone()
    for _ in range(3):
        approximator.fit(state, policy_state, lengths, target)
    assert not torch.allclose(weights_before, approximator.get_weights())

    stacked = approximator.predict(state, policy_state, lengths=lengths)
    assert stacked.shape == (3, batch)

    mean = approximator.predict(state, policy_state, lengths=lengths, prediction='mean')
    assert mean.shape == (batch,)
    assert torch.allclose(mean, stacked.mean(0))


def test_recurrent_ensemble_multi_output():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden, batch, seq = 4, 2, 6, 8, 5

    ensemble = RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=[(dim_action,), (n_hidden,)], network=RecurrentActorNetwork,
        policy_state_shape=(n_hidden,), n_models=3, n_features=8, rnn_type='gru',
        n_hidden_features=n_hidden, num_hidden_layers=1
    )

    state = torch.randn(batch, seq, dim_env)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 1, 5])

    action, next_policy_state = ensemble.predict(state, policy_state, lengths=lengths)
    assert action.shape == (batch, dim_action)
    assert next_policy_state.shape == (batch, 1, n_hidden)

    stacked_action, stacked_state = ensemble.predict(state, policy_state, lengths=lengths, prediction='all')
    assert stacked_action.shape == (3, batch, dim_action)
    assert stacked_state.shape == (3, batch, 1, n_hidden)
    assert torch.allclose(action, stacked_action.mean(0))
    assert torch.allclose(next_policy_state, stacked_state.mean(0))

    for i in range(3):
        model_action, model_state = ensemble.predict(state, policy_state, lengths=lengths, idx=i)
        assert torch.allclose(stacked_action[i], model_action)
        assert torch.allclose(stacked_state[i], model_state)

    (mean_action, var_action), _ = ensemble.predict(state, policy_state, lengths=lengths,
                                                    prediction='mean', compute_variance=True)
    assert torch.allclose(mean_action, stacked_action.mean(0))
    assert torch.allclose(var_action, stacked_action.var(0))


def test_recurrent_ensemble_multi_output_after_save_load(tmpdir):
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden, batch, seq = 4, 2, 6, 8, 5

    ensemble = RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=[(dim_action,), (n_hidden,)], network=RecurrentActorNetwork,
        policy_state_shape=(n_hidden,), n_models=3, n_features=8, rnn_type='gru',
        n_hidden_features=n_hidden, num_hidden_layers=1
    )

    state = torch.randn(batch, seq, dim_env)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 1, 5])
    expected_action, expected_state = ensemble.predict(state, policy_state, lengths=lengths)

    path = str(tmpdir / 'recurrent_multi_output_ensemble.msh')
    ensemble.save(path, full_save=True)
    loaded = RecurrentTorchApproximator.load(path)

    action, next_policy_state = loaded.predict(state, policy_state, lengths=lengths)
    assert torch.allclose(action, expected_action)
    assert torch.allclose(next_policy_state, expected_state)

    stacked_action, stacked_state = loaded.predict(state, policy_state, lengths=lengths, prediction='all')
    assert stacked_action.shape == (3, batch, dim_action)
    assert stacked_state.shape == (3, batch, 1, n_hidden)


def test_recurrent_approximator_diff_not_supported():
    torch.manual_seed(1)
    dim_env, dim_action, n_hidden = 4, 2, 6

    approximator = _make_recurrent_actor_approximator(dim_env, dim_action, n_hidden)
    ensemble = RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=(1,), network=RecurrentCriticNetwork,
        policy_state_shape=(n_hidden,), n_models=3, n_features=8, rnn_type='gru',
        n_hidden_features=n_hidden, num_hidden_layers=1
    )

    state, policy_state = torch.randn(dim_env), torch.zeros(n_hidden)
    with pytest.raises(NotImplementedError):
        approximator.diff(state, policy_state)

    with pytest.raises(NotImplementedError):
        ensemble.diff(state, policy_state)


def test_recurrent_ensemble_prediction_all():
    torch.manual_seed(1)
    dim_env, n_hidden, batch, seq = 4, 6, 8, 5

    approximator = RecurrentTorchApproximator(
        input_shape=(dim_env,), output_shape=(1,), network=RecurrentCriticNetwork,
        policy_state_shape=(n_hidden,), n_models=3, prediction='mean', n_features=8, rnn_type='gru',
        n_hidden_features=n_hidden, num_hidden_layers=1
    )

    state = torch.randn(batch, seq, dim_env)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([5, 4, 3, 5, 2, 4, 1, 5])

    mean = approximator.predict(state, policy_state, lengths=lengths)
    assert mean.shape == (batch,)

    stacked = approximator.predict(state, policy_state, lengths=lengths, prediction='all')
    assert stacked.shape == (3, batch)
    assert torch.allclose(mean, stacked.mean(0))
    for i in range(3):
        assert torch.allclose(stacked[i], approximator.predict(state, policy_state, lengths=lengths, idx=i))
