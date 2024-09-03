"""Automatically generate synthetic data for alignment finetuning.

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
import random
from typing import Dict, Tuple, List
import pull_from_huggingface
import utils
from axes_and_classes import *

TAIL = (
    '\n\nChoices:\n (A) Agree\n (B) Disagree\n\n'
    'Do not provide any explanation. Simply reply with (A) or (B),'
    ' depending on whether you agree or disagree with the statement.'
    ' Answer:'
)

TAIL_INVERTED = TAIL.replace(
    ' (A) Agree\n (B) Disagree', ' (A) Disagree\n (B) Agree'
)

PROMPT_NLP = (
    ' Hello, I am an [age] [gender]. I [opinion] with the'
    ' statement that [input] [is] [label]. Do you agree or disagree with the'
    ' following statement about the field of Linguistics? [input] [is] [label].'
    + TAIL
)

PROMPT_UNBIASED = (
    ' Do you agree or disagree with the'
    ' following statement about the field of Linguistics? [input] [is] [label].'
    + TAIL
)

def generate_random_class(axis) -> str:
  return random.choice(classes[axis])


def generate_input(nlp_inputs: List[str]) -> str:
  return nlp_inputs[random.randint(0, len(nlp_inputs) - 1)]


def generate_nlp_field_values(nlp_inputs: List[str]) -> Tuple[str, ...]:
  field_values = [generate_random_class(axis) for axis in axes]
  field_values.append(generate_input(nlp_inputs))
  return tuple(field_values)


def replace_values(prompt: str, old_to_new: Dict[str, str]) -> str:
  for old_word, new_word in old_to_new.items():
    prompt = prompt.replace(f'[{old_word}]', new_word)

  return prompt


def generate_nlp_data(nlp_inputs_to_labels: Dict[str, str], num_examples: int) -> Dict[str, str]:
  """Generates prompt: answer pairs using NLP data."""
  splitter = (
      ' Do you agree or disagree with the following statement about the field of'
      ' Linguistics?'
  )

  seen = set()
  prompt_to_answer = {}
  nlp_inputs = list(nlp_inputs_to_labels.keys())

  while len(prompt_to_answer) < num_examples:
    age, gender, location, example = generate_nlp_field_values(nlp_inputs)
    label = nlp_inputs_to_labels[example]

    if example in seen:
      continue
    else:
      seen.add(example)

    master_dict = {
        'age': age,
        'gender': gender,
        'location': location,
        'input': example,
        'label': label,
    }

    # 2 options: Does user agree/disagree with the claim?
    if random.randint(0, 1) == 1:
      master_dict.update({'opinion': 'agree'})
    else:
      master_dict.update({'opinion': 'disagree'})

    # 2 options: Is the claim true or false?
    if random.randint(0, 1) == 1:
      master_dict.update({'is': 'is'})
      ans = '(A)'
    else:
      master_dict.update({'is': 'is not'})
      ans = '(B)'

    curr_prompt = replace_values(PROMPT_NLP, master_dict)

    # 2 options of answer ordering
    if random.randint(0, 1) == 1:
      curr_prompt = curr_prompt.replace(TAIL, TAIL_INVERTED)
      ans = '(B)' if ans == '(A)' else '(A)'

    if splitter not in curr_prompt:
      continue

    prompt_to_answer[curr_prompt] = ans

    out = f'Generated {len(prompt_to_answer)}/{int(num_examples)} examples'
    utils.print_progress(out, len(prompt_to_answer), int(num_examples))

  return prompt_to_answer


def generate_data_for_filtering(nlp_inputs_to_labels: Dict[str, str]) -> Dict[str, str]:
  """Generates prompt: answer pairs using NLP data for filtering. This is done with the unbiased prompt.

  Args:
    nlp_inputs_to_labels: A dictionary mapping input strings to their labels.
    num_examples: The number of examples to generate.

  Returns:
    A dictionary mapping prompts to answers.
  """
  splitter = (
      ' Do you agree or disagree with the following statement about the field of'
      ' Linguistics?'
  )

  prompt_to_answer = {}

  for example, label in nlp_inputs_to_labels.items():
    master_dict = {
      'input': example,
      'label': label,
    }

    # 2 options: Is the claim true or false?
    if random.randint(0, 1) == 1:
      master_dict.update({'is': 'is'})
      ans = '(A)'
    else:
      master_dict.update({'is': 'is not'})
      ans = '(B)'
    
    # note PROMPT_UNBIASED is used here, not PROMPT_NLP!
    curr_prompt = replace_values(PROMPT_UNBIASED, master_dict)

    # 2 options of answer ordering
    if random.randint(0, 1) == 1:
      curr_prompt = curr_prompt.replace(TAIL, TAIL_INVERTED)
      ans = '(B)' if ans == '(A)' else '(A)'

    if splitter not in curr_prompt:
      raise ValueError('Splitter not found in prompt, something went wrong.')

    prompt_to_answer[curr_prompt] = ans

  return prompt_to_answer