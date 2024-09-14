import pandas as pd
import pull_from_huggingface
import random
import generate_data as gd

NLP_INPUTS_TO_LABELS = pull_from_huggingface.collect_all_datasets();
print(f'#Downloaded samples: {len(NLP_INPUTS_TO_LABELS)}')

# Choose a subset of the loaded data
def create_data_subset() -> pd.DataFrame:
    print(f'Creating a subset of {N_STATEMENTS_TO_FILTER} NLP statements...')
    NLP_INPUTS_TO_LABELS_SUBSET = dict(random.sample(list(NLP_INPUTS_TO_LABELS.items()), N_STATEMENTS_TO_FILTER))

    df_subset = pd.DataFrame.from_dict(NLP_INPUTS_TO_LABELS_SUBSET, orient='index', columns=['label'])
    df_subset.reset_index(inplace=True)
    df_subset.columns = ['input', 'label']

    df_subset.to_csv('data_storage/input_label_pairs_unfiltered.csv', index=False)

    return df_subset

# Filter out pairs for which the model does not know the answer
def filter_data(df_unfiltered: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out pairs for which the model does not know the answer
    """
    print('Filtering out statements for which the model does not know the answer...')
    if 'input' not in df_unfiltered.columns or 'label' not in df_unfiltered.columns:
        raise ValueError("DataFrame must contain 'input' and 'label' columns")

    dict_unfiltered = gd.generate_data_for_filtering(dict(zip(df_unfiltered['input'], df_unfiltered['label'])))
    can_be_used = []

    responses = []
    for prompt, truthful_answer in dict_unfiltered.items():
        response = get_completion(prompt, model=MODEL)
        responses.append(response)

        if response != truthful_answer:
            can_be_used.append(False)
        else:
            can_be_used.append(True)

    print(f'{can_be_used.count(True) / len(can_be_used) * 100:.2f}% of the data can be used')

    df_filtered = df_unfiltered.copy()
    # Add these three columns as a sanity check, so that we can compare the truthful and actual answers
    df_filtered['prompt'] = list(dict_unfiltered.keys())
    df_filtered['truthful_answer'] = list(dict_unfiltered.values())
    df_filtered['actual_answer'] = responses
    # Filter out the data
    df_filtered = df_filtered[can_be_used]
    df_filtered.to_csv('data_storage/input_label_pairs_filtered.csv', index=False)

    #train/test split
    df_train = df_filtered.sample(frac=0.8, random_state=42)
    df_test = df_filtered.drop(df_train.index)

    df_train.to_csv('data_storage/input_label_pairs_filtered_train.csv', index=False)
    df_test.to_csv('data_storage/input_label_pairs_filtered_test.csv', index=False)

    return df_train, df_test, df_filtered


if __name__ == '__main__':
    PROVIDER = 'openai'
    MODEL = 'gpt-4o-mini-2024-07-18'

    if PROVIDER == 'openai':
        from openai_interface import get_completion
    else:
        raise ValueError(f'Unknown provider: {PROVIDER}')

    N_STATEMENTS_TO_FILTER = 100
    df_subset = create_data_subset()
    df_train, df_test, df_filtered = filter_data(df_subset)
else:
    from config import *
