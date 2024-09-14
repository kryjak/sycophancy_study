N_STATEMENTS_TO_FILTER = 30
NUM_EXAMPLES_TRAIN = 100
NUM_EXAMPLES_TEST = 20
N_OPENENDED_STATEMENTS = 30
N_OPENENDED_EXAMPLES = 20

PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'
WANDB_INTEGRATION = True

if PROVIDER == 'openai':
    from openai_interface import *
    from openai_finetuning_config import *
    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION)
else:
    raise ValueError(f'Unknown provider: {PROVIDER}')