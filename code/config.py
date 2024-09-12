N = 25 # choose how many NLP statements to use for filtering
PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'
WANDB_INTEGRATION = True

if PROVIDER == 'openai':
    from openai_interface import *
    from openai_finetuning_config import *
    finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION)
else:
    raise ValueError(f'Unknown provider: {PROVIDER}')