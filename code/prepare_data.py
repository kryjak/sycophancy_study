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
Create prompts for fine-tuning and experiments.
"""

import pandas as pd
from axes_and_classes import *
import generate_data as gd
import json

NLP_STATEMENTS_OPENENDED: list[str] = json.load(open('nlp_statements_openended.json'))

def create_prompts(df_train: pd.DataFrame, df_test: pd.DataFrame, axes: list[str], num_examples_train: int, num_examples_test: int) -> None:
    print('Generating prompts for the closed-ended experiments...')

    if num_examples_train > len(df_train):
        num_examples_train = len(df_train)
        print(f'num_examples_train is greater than the number of training examples, so we set it to {num_examples_train}')
    if num_examples_test > len(df_test):
        num_examples_test = len(df_test)
        print(f'num_examples_test is greater than the number of test examples, so we set it to {num_examples_test}')

    # Convert to dictionaries because `gd.generate_nlp_data_easy` expects a dictionary:
    dict_train = dict(zip(df_train['input'], df_train['label']))
    dict_test = dict(zip(df_test['input'], df_test['label']))

    for axis in axes:
        print(f'Generating prompts for {axis}...')
        train_prompts_df = gd.generate_nlp_data_easy(dict_train, axis, num_examples_train)
        test_prompts_easy_df = gd.generate_nlp_data_easy(dict_test, axis, num_examples_test)
        test_prompts_hard_df = gd.generate_nlp_data_hard(dict_test, axis, num_examples_test)

        train_prompts_df.to_csv(f'data_storage/train_prompts_{axis}.csv', index=False)
        test_prompts_easy_df.to_csv(f'data_storage/test_prompts_easy_{axis}.csv', index=False)
        test_prompts_hard_df.to_csv(f'data_storage/test_prompts_hard_{axis}.csv', index=False)

def create_prompts_openended_unbiased(num_examples: int) -> pd.DataFrame:
    if num_examples > len(NLP_STATEMENTS_OPENENDED):
        num_examples = len(NLP_STATEMENTS_OPENENDED)
        print(f'num_examples is greater than the number of open-ended statements, so we set it to {num_examples}')

    df = gd.generate_nlp_data_openended_unbiased(NLP_STATEMENTS_OPENENDED, num_examples)
    df.to_csv('data_storage/test_prompts_openended_unbiased.csv', index=False)
    return df

def create_prompts_openended_biased(df: pd.DataFrame, axes: list[str], num_examples: int) -> pd.DataFrame:
    print('Generating prompts for the open-ended experiments...')
    if num_examples > len(df):
        print(f'Careful, num_examples ({num_examples}) is greater than the number of open-ended statements ({len(df)})!')
        print('You might run out of combinations of statements and axes.')

    nlp_statements_openeded = df['statement'].tolist()

    for axis in axes:
        test_prompts_openended_df = gd.generate_nlp_data_openended(nlp_statements_openeded, axis, num_examples)
        test_prompts_openended_df.to_csv(f'data_storage/test_prompts_openended_{axis}.csv', index=False)
    return test_prompts_openended_df

if __name__ == '__main__':
    NUM_EXAMPLES_TRAIN = 100
    NUM_EXAMPLES_TEST = 20
    N_OPENENDED_STATEMENTS = 32
    N_OPENENDED_EXAMPLES = 100

    df_train = pd.read_csv('data_storage/input_label_pairs_filtered_train.csv')
    df_test = pd.read_csv('data_storage/input_label_pairs_filtered_test.csv')
    df_openended = create_prompts_openended_unbiased(N_OPENENDED_STATEMENTS)

    if 'input' not in df_train.columns or 'label' not in df_train.columns:
        raise ValueError("Train DataFrame must contain 'input' and 'label' columns")
    if 'input' not in df_test.columns or 'label' not in df_test.columns:
        raise ValueError("Test DataFrame must contain 'input' and 'label' columns")

    create_prompts(df_train, df_test, axes, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST)
    create_prompts_openended_biased(df_openended, axes, N_OPENENDED_EXAMPLES)
