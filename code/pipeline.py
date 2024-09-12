"""Pipeline for generating synthetic tuning/evaluation data.

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

import axes_and_classes as ac
import download_and_filter_data as dfd
import prepare_data as pd
import fine_tune as ft
import run_experiments as re

# -----------SETTINGS-----------
N = 100 # choose how many NLP statements to use for filtering
PROVIDER = 'openai'
MODEL = 'gpt-4o-mini-2024-07-18'
WANDB_INTEGRATION = True

if PROVIDER == 'openai':
    from openai_interface import get_completion
    from openai_finetuning_config import *
else:
    raise ValueError(f'Unknown provider: {PROVIDER}')

# -----------DATASET PIPELINE-----------
print(f'Axes used: {ac.axes}')
print(f'Their classes: {ac.classes}')
print(f'...and affirmative class: {ac.affirmative_class}')

print(f'Creating a subset of {N} NLP statements...')
df_subset = dfd.create_data_subset(N)
print(f'Filtering out statements for which the model does not know the answer...')
_ = dfd.filter_data(df_subset)
print(f'Filtering complete.')

print(f'Generating prompts for fine-tuning and experiments...')
pd.create_prompts(ac.axes)
print(f'Prompts generated.')

print(f'Submitting fine-tuning jobs...')
ft.submit_fine_tuning_jobs(ac.axes)
print(f'All fine-tuning jobs submitted.')

print(f'Running experiments...')
n_jobs = len(ac.axes)
fine_tuned_models = re.wait_for_fine_tuning_jobs(n_jobs)
assert len(fine_tuned_models) == n_jobs, f'Expected {n_jobs} fine-tuned models, but got {len(fine_tuned_models)}'
re.run_experiments(fine_tuned_models)
print(f'All experiments completed.')

