"""Automatically generate synthetic data for alignment finetuning.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Create the fine-tuning configuration based on pre-defined parameters.
"""

import os
from typing import Optional

def get_finetuning_config(model: Optional[str] = 'gpt-4o-mini-2024-07-18', wandb_integration: bool = False, hyperparameters: dict = None) -> dict:
    finetuning_config = {
        'model': model,
        'seed': 42
        }

    if hyperparameters != None:
        finetuning_config['hyperparameters'] = hyperparameters

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
    N_EPOCHS = 1
    BATCH_SIZE = 8
    LEARNING_RATE_MULTIPLIER = 1

    hyperparameters = {
        'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate_multiplier': LEARNING_RATE_MULTIPLIER
    }

    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION, hyperparameters)
    print(finetuning_config)
