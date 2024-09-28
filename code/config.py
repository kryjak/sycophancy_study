"""
Copyright 2023 Jakub Kryś

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License (CC BY-NC 4.0). To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/

This work is based on original work by Google LLC, licensed under the Apache 
License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

See the LICENSE file in the project root for full license information.
"""

"""
Master settings for running fine-tuning and experiments.
"""

N_STATEMENTS_TO_FILTER = 10000
NUM_EXAMPLES_TRAIN = 5000
NUM_EXAMPLES_TEST = 1000
N_OPENENDED_STATEMENTS = 32
N_OPENENDED_EXAMPLES = 100

PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'
WANDB_INTEGRATION = True

# FINE_TUNING CONFIG
N_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE_MULTIPLIER = 1

if PROVIDER == 'openai':
    from openai_interface import *
    from openai_finetuning_config import *
    hyperparameters = {
        'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate_multiplier': LEARNING_RATE_MULTIPLIER
    }
    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION, hyperparameters)
else:
    raise ValueError(f'Unknown provider: {PROVIDER}')
