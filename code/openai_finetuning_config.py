import os
from typing import Optional

def get_finetuning_config(model: Optional[str] = 'gpt-4o-mini-2024-07-18', wandb_integration: bool = False) -> dict:
    finetuning_config = {
        'model': model,
        'hyperparameters': {
            'n_epochs': 1,
            'batch_size': 'auto',
            'learning_rate_multiplier': 'auto',
        },
        'seed': 42
    }

    if wandb_integration:
        WANDB_PROJECT = 'sycophancy_study'
        WANDB_ENTITY = os.environ['WANDB_ENTITY']

        finetuning_config['integrations'] = [{
            'type': 'wandb',
            'wandb': {
                'project': WANDB_PROJECT,
                'entity': WANDB_ENTITY
            }
        }]
    return finetuning_config

if __name__ == '__main__':
    MODEL = 'gpt-4o-mini-2024-07-18'
    WANDB_INTEGRATION = True
    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION)
