# Not all biases are equal -- a study of sycophancy and bias in LLMs

Code for my [AI Safety Fundamentals](https://aisafetyfundamentals.com/alignment/) project on introducing sycophancy and biases into LLMs through synthetic data fine-tuning. Project report can be found at `main.pdf`.

## Code
Running the code requires the `datasets` package. To replicate our results, after cloning execute:
```
cd code
python pipeline.py
```
###


### File description
Main pipeline:
- `pipeline.py` -- main file that integrates the full pipeline.

The full pipeline, with all parameters set to default values, should cost around $25 to run. Its components can be executed separately to complete individual subtasks:
- download_and_filter_data.py --  downloads datasets from HuggingFace and filters out statements that the model is not able to correctly answer.
- prepare_data.py -- generate prompts for fine-tuning and experiments
- fine_tune.py -- submit fine-tuning jobs through an API
- run_experiments.py -- run experiments on the original and fine-tuned models
- analyse_results.py -- extract information from the results and create plots
 
 Other files:
- config.py -- main config file to specify parameters such as the model, number of prompts, batch size, [W&B](https://wandb.ai/site) integration, etc.
- [provider]_interface.py -- functions to interact with the API of a given provided such as OpenAI
- [provider]_finetuning_config.py -- file to create config for fine-tuning jobs from pre-defined parameters
- axes_and_classes.py -- a list of axes and classes used in the study (see the report)
- experiment_list.py -- a list of experiments to test the models on
- pull_from_huggingface.py -- functions to pull datasets from HuggingFace
- utils.py -- utility functions

Finally, the open-ended NLP statements of [Michael et al., 2022](https://arxiv.org/abs/2208.12852) are collected for convenience in `nlp_statements_openended.json`.
## License

This project contains code licensed under the Apache License 2.0 and original contributions licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

- This repository is a fork of [https://github.com/google/sycophancy-intervention/](https://github.com/google/sycophancy-intervention/) by Google LLC, which was licensed under Apache License 2.0. See [LICENSE-APACHE](https://www.apache.org/licenses/LICENSE-2.0.html) for details.
- Original contributions and modifications by Jakub Kryś are licensed under CC BY-NC 4.0. See [LICENSE-CC-BY-NC](https://www.creativecommons.org/licenses/by-nc/4.0/legalcode.en) for details.

For a list of changes made to the original repository, see `NOTICES.md`.

## Contributing

Feel free to clone, fork or open an issue. Code subject to the CC BY-NC 4.0 license.

## Contact

Jakub Kryś

jakkrys@gmail.com