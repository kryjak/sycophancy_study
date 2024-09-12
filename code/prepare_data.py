import pandas as pd
from axes_and_classes import *
import generate_data as gd

def create_prompts(df_train: pd.DataFrame, df_test: pd.DataFrame, axes: list[str]) -> None:
    # Convert to dictionaries because `gd.generate_nlp_data_easy` expects a dictionary:
    dict_train = dict(zip(df_train['input'], df_train['label']))
    dict_test = dict(zip(df_test['input'], df_test['label']))

    NUM_EXAMPLES_TRAIN = len(dict_train)
    NUM_EXAMPLES_TEST = len(dict_test)

    for axis in axes:
        print(f'Generating prompts for {axis}...')
        train_prompts_df = gd.generate_nlp_data_easy(dict_train, axis, NUM_EXAMPLES_TRAIN)
        test_prompts_easy_df = gd.generate_nlp_data_easy(dict_test, axis, NUM_EXAMPLES_TEST)
        test_prompts_hard_df = gd.generate_nlp_data_hard(dict_test, axis, NUM_EXAMPLES_TEST)

        train_prompts_df.to_csv(f'data_source_nlp/train_prompts_{axis}.csv', index=False)
        test_prompts_easy_df.to_csv(f'data_source_nlp/test_prompts_easy_{axis}.csv', index=False)
        test_prompts_hard_df.to_csv(f'data_source_nlp/test_prompts_hard_{axis}.csv', index=False)

if __name__ == '__main__':
    df_train = pd.read_csv('data_source_nlp/input_label_pairs_filtered_train.csv')
    df_test = pd.read_csv('data_source_nlp/input_label_pairs_filtered_test.csv')

    if 'input' not in df_train.columns or 'label' not in df_train.columns:
        raise ValueError("Train DataFrame must contain 'input' and 'label' columns")
    if 'input' not in df_test.columns or 'label' not in df_test.columns:
        raise ValueError("Test DataFrame must contain 'input' and 'label' columns")

    print(df_train.head())
    print(df_test.head())

    create_prompts(df_train, df_test, axes)