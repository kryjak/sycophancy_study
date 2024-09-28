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
Functions for downloading datasets from HuggingFace.
"""

import os
import random
from typing import List, Tuple, Dict
import datasets
import utils

DATA_FOLDER = 'data_storage'
DATASET_NAMES = [
    'rotten_tomatoes',
    'tweet_eval',
    'glue',
    'glue',
    'glue',
]
DATASET_SUBSETS = [
    '',
    'irony',
    'mrpc',
    'wnli',
    'cola',
]
DATASET_LABELS = [
    ['Negative Sentiment', 'Positive Sentiment'],
    ['Not Irony', 'Irony'],
    ['Not Equivalent', 'Equivalent'],
    ['Not Entailment', 'Entailment'],
    ['Unacceptable Sentence', 'Acceptable Sentence'],
]
DATASET_FIELDS = [
    ['text'],
    ['text'],
    ['sentence1', 'sentence2'],
    ['sentence1', 'sentence2'],
    ['sentence'],
]
DATASET_LABEL_NAMES = [
    'label',
    'label',
    'label',
    'label',
    'label',
]


def clean_input(example: str) -> str:
  example = example.replace('"', '')
  example = example.replace('\n', '')
  example = example.strip()
  example = f'"{example}"'
  return example


def build_dataset(
    dataset_name: str,
    dataset_subset: str,
    label_name: str,
    text_fields: List[str],
    natural_language_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
  """Uses inputted dataset details to build dictionary of train/test values."""
  if not dataset_subset:
    dataset_dict = datasets.load_dataset(dataset_name)
  else:
    dataset_dict = datasets.load_dataset(dataset_name, name=dataset_subset)

  train_dict, test_dict = {}, {}
  has_validation = False

  if 'validation' in dataset_dict:
    train_data, test_data = dataset_dict['train'], dataset_dict['validation']
    has_validation = True
  elif 'test' in dataset_dict:
    train_data, test_data = dataset_dict['train'], dataset_dict['test']
  else:
    print('WARNING: NO EXISTING SPLITS FOUND')
    temp = list(dataset_dict['train'])
    random.shuffle(temp)
    num_test_examples = int(.2 * len(temp))
    train_data, test_data = temp[:-num_test_examples], temp[-num_test_examples:]

  for data, dict_ in zip([train_data, test_data], [train_dict, test_dict]):
    for i in range(len(data)):
      text = ' and '.join([clean_input(data[i][x]) for x in text_fields])
      label = data[i][label_name]
      dict_[text] = natural_language_labels[label]

  print(f'Found {len(train_dict)} training examples')
  print(f'Found {len(test_dict)} testing examples')

  return train_dict, test_dict, has_validation


def build_all_datasets() -> None:
  """Builds all input:label dictionaries and saves them."""
  for name, subset, labels, fields, label_name in zip(
      DATASET_NAMES,
      DATASET_SUBSETS,
      DATASET_LABELS,
      DATASET_FIELDS,
      DATASET_LABEL_NAMES,
  ):
    full_name = name if not subset else f'{name}_{subset}'
    out_path = os.path.join(DATA_FOLDER, full_name + '.pickle')

    if not os.path.exists(DATA_FOLDER):
      os.mkdir(DATA_FOLDER)

    # Do not reprocess datasets
    if os.path.exists(out_path):
      continue

    train_dict, _, _ = build_dataset(name, subset, label_name, fields, labels)
    # utils.print_an_example(train_dict)

    utils.save_pickle(out_path, train_dict)


def collect_all_datasets() -> Dict[str, str]:
  build_all_datasets()

  result = {}

  for name, subset in zip(DATASET_NAMES, DATASET_SUBSETS):
    full_name = name if not subset else f'{name}_{subset}'
    out_path = os.path.join(DATA_FOLDER, full_name + '.pickle')
    result.update(utils.load_pickle(out_path))

  return result
