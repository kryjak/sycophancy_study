N_STATEMENTS_TO_FILTER = 10000
NUM_EXAMPLES_TRAIN = 5000
NUM_EXAMPLES_TEST = 1000
N_OPENENDED_STATEMENTS = 32
N_OPENENDED_EXAMPLES = 100

PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'
WANDB_INTEGRATION = True

if PROVIDER == 'openai':
    from openai_interface import *
    from openai_finetuning_config import *
    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION)
else:
    raise ValueError(f'Unknown provider: {PROVIDER}')