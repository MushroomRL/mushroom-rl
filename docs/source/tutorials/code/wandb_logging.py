from mushroom_rl.core import Logger

# Build the wandb init arguments, including the experiment hyperparameters
hyperparams = dict(gamma=0.99, lr=3e-4, batch_size=64)
wandb_kwargs = Logger.default_wandb_kwargs('tutorial_project',
                                           config=hyperparams,
                                           name='tutorial_run',
                                           mode='offline')

# Create a logger with wandb logging enabled
logger = Logger('wandb_tutorial', results_dir='/tmp/logs',
                wandb_kwargs=wandb_kwargs, force_numpy=False)

# Log some metrics to every active backend
logger.log(actor_loss=0.5, critic_loss=1.2)

# Advance the wandb step and log the next values
logger.advance_step()
logger.log(actor_loss=0.4, critic_loss=1.0)