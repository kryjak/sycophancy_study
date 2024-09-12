"""
Create data for fine-tuning and submit a fine-tuning job the the API.
For OpenAI fine-tuning API, the required format is:
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
The file with such dictionaries should be saved in .jsonl format.
"""
import pandas as pd
import os
from axes_and_classes import axes
from sklearn.model_selection import train_test_split
# submit fine-tuning jobs for each axis
def submit_fine_tuning_jobs(axes: list[str], finetuning_config: dict) -> None:
    for axis in axes:
        train_prompts_df = pd.read_csv(f'data_storage/train_prompts_{axis}.csv')
        # train/validation split
        train_prompts_df, validation_prompts_df = train_test_split(train_prompts_df, test_size=0.2, random_state=42)
        # create fine-tuning data
        create_fine_tuning_data(train_prompts_df, f'data_storage/fine_tuning_data_{axis}.jsonl')
        create_fine_tuning_data(validation_prompts_df, f'data_storage/fine_tuning_data_{axis}_validation.jsonl')
        # upload files to OpenAI
        train_file_id = upload_files(f'data_storage/fine_tuning_data_{axis}.jsonl')
        validation_file_id = upload_files(f'data_storage/fine_tuning_data_{axis}_validation.jsonl')
        print(f'Number of training samples: {len(train_prompts_df)}')
        print(f'Number of validation samples: {len(validation_prompts_df)}')
        # submit fine-tuning job
        job_id = submit_fine_tuning_job(
            training_file=train_file_id,
            validation_file=validation_file_id,
            suffix=f'{axis}_finetuned',
            **finetuning_config
        )
        print(f'Fine-tuning job submitted: {job_id}')

if __name__ == '__main__':
    PROVIDER = 'openai'
    MODEL = 'gpt-4o-mini-2024-07-18'
    WANDB_INTEGRATION = True

    if PROVIDER == 'openai':
        from openai_interface import *
        from openai_finetuning_config import *
        finetuning_config = get_finetuning_config(MODEL, WANDB_INTEGRATION)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")

    submit_fine_tuning_jobs(axes, finetuning_config)
