from mushroom_rl.core import Logger

# Build the wandb init arguments, including the experiment hyperparameters
hyperparams = dict(gamma=0.99, lr=3e-4, batch_size=64)
wandb_kwargs = Logger.default_wandb_kwargs('tutorial_project',
                                           config=hyperparams,
                                           name='tutorial_run',
                                           mode='offline')

# Create a logger with wandb logging enabled
logger = Logger('wandb_tutorial', results_dir='/tmp/logs', use_timestamp=True,
                wandb_kwargs=wandb_kwargs, force_numpy=False)

# Log training metrics, grouped under training/ and using n_fit as x-axis
logger.log_training(**{'actor/loss': 0.5, 'critic/loss': 1.2})

# Advance the number of fits counter once per fit
logger.advance_step()
logger.log_training(**{'actor/loss': 0.4, 'critic/loss': 1.0})

# Log evaluation metrics, grouped under eval/ and using the epoch as x-axis
logger.log_evaluation(0, J=10.0, R=20.0)
