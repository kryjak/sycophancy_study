# Notice of changes to the original repository
This repository is a fork of [https://github.com/google/sycophancy-intervention/](https://github.com/google/sycophancy-intervention/) by Google LLC, which was licensed under Apache License 2.0. See [LICENSE-APACHE](https://www.apache.org/licenses/LICENSE-2.0.html) for details.

Whilst the majority of the code in this repository is original, we have retained some prior code and implemented the following key changes:
- `generate_data.py` -- this is the most important change. Because we *introduce* rather than *reduce* sycophancy, the way we construct prompts is very different. In particular, we artificially correlate the sycophantic and truthful answers (see the report for details). The functions in this file should be used with care, otherwise they will lead to mistakes in downstream tasks.
- `pull_from_huggingface.py` -- we reduced the number of datasets downloaded to just 5 (see the report.)

Also, in `utils.py` we adapted the `print_progress` funtion to make the progress bar display correctly in Jupyter notebooks.