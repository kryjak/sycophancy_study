"""
Retrieve the fine-tuning job status and results.
Run all experiments and evaluate the fine-tuned models.
Aggregate the results and plot the performance of each model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from axes_and_classes import axes
from experiment_list import experiments
from typing import List

PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'

if PROVIDER == 'openai':
    from openai_interface import *
else:
    raise ValueError(f"Unknown provider: {PROVIDER}")

n_jobs = len(axes)

# list all fine-tuning jobs:
print('Listing all fine-tuning jobs:')
print(list_fine_tuning_jobs())

# wait if all jobs are completed:
while True:
    print('Waiting for all fine-tuning jobs to complete...')
    time.sleep(60)
    
    jobs = list_fine_tuning_jobs(n_jobs)
    
    all_completed = True
    for job in jobs:
        if job.get('fine_tuned_model') is None:
            all_completed = False
            break
    
    if all_completed:
        print('All fine-tuning jobs are completed!')
        break

# retrieve fine-tuned model names:
print('Retrieving fine-tuned model names:')
fine_tuned_models = [job.get('fine_tuned_model') for job in jobs]
print(f'Fine-tuned models: {fine_tuned_models}')

### RUN EXPERIMENTS ###
def run_experiment(df: pd.DataFrame, experiment: str, axis: str, fine_tuned_models: List[str]) -> pd.DataFrame:
    suffix = f'{axis}_finetuned'
    # select the fine-tuned model for the relevant axis:
    fine_tuned_model = [model for model in fine_tuned_models if suffix in model][0]

    for index, row in df.iterrows():
        prompt = row['prompt']

        baseline_answer = get_completion(prompt, model=MODEL)
        df.at[index, 'baseline_answer'] = baseline_answer

        answer = get_completion(prompt, model=fine_tuned_model)
        df.at[index, 'actual_answer'] = answer

    df.to_csv(f'data_source_nlp/results_{experiment}_{axis}.csv', index=False)
    return df

print(f'Experiments to be run: {experiments}')
print(f'Axes to be run: {axes}')
print('Starting experiments...')
print('-' * 80)

for experiment in experiments:
    for axis in axes:
        print(f'Running experiment {experiment} for axis {axis}...')
        df = pd.read_csv(f'data_source_nlp/test_prompts_{experiment}_{axis}.csv')
        df = run_experiment(df, experiment, axis, fine_tuned_models)

# Also test the filtered out statements (without user's claims) on fine-tuned models.
# This is to check if fine-tuning affects the underlying knowledge.
# This is an equivalent experiment to Appendix A.4 of [2308.03958].
print(f'Now checking if fine-tuning affects the underlying knowledge...')
for axis in axes:
    for case in ['train', 'test']:  
        suffix = f'{axis}_finetuned'
        fine_tuned_model = [model for model in fine_tuned_models if suffix in model][0]

        df = pd.read_csv(f'data_source_nlp/inputs_label_pairs_filtered_{case}.csv')
        df[f'{axes}_finetuned_answer'] = df.apply(lambda row: get_completion(row['prompt'], model=fine_tuned_model), axis=1)
        df.to_csv(f'data_source_nlp/filtering_knowledge_check_{case}_{axis}.csv', index=False)

print('-' * 80)
print('Experiments completed!')
