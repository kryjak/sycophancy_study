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
Main pipeline for running data filtering, prompt creation, fine-tuning, experiments, analysis and plotting.
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

df_subset = dfd.create_data_subset()
df_train, df_test, _ = dfd.filter_data(df_subset)
print('Filtering complete.')

print('Generating prompts for fine-tuning and experiments...')
pd.create_prompts(df_train, df_test, ac.axes, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST)
df_openended = pd.create_prompts_openended_unbiased(N_OPENENDED_STATEMENTS)
pd.create_prompts_openended_biased(df_openended, ac.axes, N_OPENENDED_EXAMPLES)
print('All prompts generated.')

ft.submit_fine_tuning_jobs(ac.axes)

print('Running experiments...')
n_jobs = len(ac.axes)
fine_tuned_models = re.wait_for_fine_tuning_jobs(n_jobs)
assert len(fine_tuned_models) == n_jobs, f'Expected {n_jobs} fine-tuned models, but got {len(fine_tuned_models)}'
re.run_all_experiments(fine_tuned_models)
print('All experiments completed.')

print('Analyzing results...')
# for experiment in exp.experiments:
#     _ = ar.create_experiment_plot(experiment)
# #     plt.show()

_ = ar.create_combined_experiment_plot()
# plt.show()

for case in ['train', 'test']:
    _ = ar.create_knowledge_check_plot(case)
    # plt.show()

_ = ar.create_openended_unbiased_sycophancy_plot()
# plt.show()
print('Results analysed.')

print('Pipeline completed.')
