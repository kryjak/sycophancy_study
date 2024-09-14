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
import matplotlib.pyplot as plt

import axes_and_classes as ac
import download_and_filter_data as dfd
import prepare_data as pd
import fine_tune as ft
import run_experiments as re
import analyse_results as ar
import experiment_list as exp

# -----------DATASET PIPELINE-----------
print(f'Axes used: {ac.axes}')
print(f'Their classes: {ac.classes}')
print(f'...and affirmative class: {ac.affirmative_class}')

df_subset = dfd.create_data_subset(N_STATEMENTS_TO_FILTER)
df_train, df_test, _ = dfd.filter_data(df_subset)
print('Filtering complete.')

print('Generating prompts for fine-tuning and experiments...')
pd.create_prompts(df_train, df_test, ac.axes, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST)
df_openended = pd.create_prompts_openended_unbiased(N_OPENENDED_STATEMENTS)
pd.create_prompts_openended_biased(df_openended, ac.axes, N_OPENENDED_EXAMPLES)
print('All prompts generated.')

ft.submit_fine_tuning_jobs(ac.axes, finetuning_config)

print('Running experiments...')
n_jobs = len(ac.axes)
fine_tuned_models = re.wait_for_fine_tuning_jobs(n_jobs)
assert len(fine_tuned_models) == n_jobs, f'Expected {n_jobs} fine-tuned models, but got {len(fine_tuned_models)}'
re.run_all_experiments(fine_tuned_models)
print('All experiments completed.')

print('Analyzing results...')
for experiment in exp.experiments:
    fig = ar.create_experiment_plot(experiment)
    plt.show()

for case in ['train', 'test']:
    fig = ar.create_knowledge_check_plot(case)
    plt.show()
print('Results analysed.')

print('Pipeline completed.')
