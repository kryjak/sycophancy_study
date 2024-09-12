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

# -----------SETTINGS-----------
from config import *

# -----------IMPORTS-----------
import axes_and_classes as ac
import download_and_filter_data as dfd
import fine_tune as ft
import run_experiments as re

# -----------DATASET PIPELINE-----------
print(f'Axes used: {ac.axes}')
print(f'Their classes: {ac.classes}')
print(f'...and affirmative class: {ac.affirmative_class}')

print('Creating a subset of {N} NLP statements...')
df_subset = dfd.create_data_subset(N)
print('Filtering out statements for which the model does not know the answer...')
df_train, df_test, _ = dfd.filter_data(df_subset)
print('Filtering complete.')

import prepare_data as pd
print('Generating prompts for fine-tuning and experiments...')
pd.create_prompts(df_train, df_test, ac.axes)
print('All prompts generated.')

print('Submitting fine-tuning jobs...')
ft.submit_fine_tuning_jobs(ac.axes, finetuning_config)
print('All fine-tuning jobs submitted.')

print('Running experiments...')
n_jobs = len(ac.axes)
fine_tuned_models = re.wait_for_fine_tuning_jobs(n_jobs)
assert len(fine_tuned_models) == n_jobs, f'Expected {n_jobs} fine-tuned models, but got {len(fine_tuned_models)}'
re.run_all_experiments(fine_tuned_models)
print('All experiments completed.')

