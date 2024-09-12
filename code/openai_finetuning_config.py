import os

if __name__ == '__main__':
    MODEL = 'gpt-4o-mini-2024-07-18'
    WANDB_INTEGRATION = True

finetuning_config = {
    'model': MODEL,
    'hyperparameters': {
        'n_epochs': 1,
        'batch_size': 'auto',
        'learning_rate_multiplier': 'auto',
    },
    'seed': 42
}

if WANDB_INTEGRATION:
    WANDB_PROJECT = 'sycophancy_study'
    WANDB_ENTITY = os.environ['WANDB_ENTITY']

    finetuning_config['integrations'] = {
        'type': 'wandb',
        'wandb': {
            'project': WANDB_PROJECT,
            'entity': WANDB_ENTITY
        }
    }