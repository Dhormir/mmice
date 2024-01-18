# Multilingual Minimal Contrastive Editing (MMiCE) 🐭

This repository contains the multilingual version code of the paper, [Explaining NLP Models via Minimal Contrastive Editing (MiCE)](https://arxiv.org/pdf/2012.13985.pdf), adapted to pytorch + accelerator for better long-term compatibility and reproducibility, plus we eliminated the now deprecated allennlp and allennlp-models package.

## Requirements

This programs was developed on windows 10 OS, using pytorch, transformers, nltk, and the common python libraries.
Currently we are working on adapting the code to unix environments.
The code development was centered around GPU for efficient execution time, although CPU only use is allowed, as of 2023 the authors do not find this to be an optimum approach.

## Authors

- Domingo Benoit Cea, Msc. Universidad Técnica Federico Santa María

## Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/allenai/mice.git
    cd mice
    ```

2. Create a python virtual environment on python>=3.10.
    for more info follow [link](https://docs.python.org/3/library/venv.html)

    ```bash
    python -m venv .env
    ```

3. Activate the environment.

    ```bash
    source .env/bin/activate
    ```

4. Download the requirements.

    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

1. **Download Task Data**: If you want to work with the RACE dataset, download it here: [Link](https://www.cs.cmu.edu/~glai1/data/race/).
The commands below assume that this data, after downloaded, is stored in `data/RACE/`.
All other task-specific datasets are automatically downloaded by the commands below.
2. **Download Pretrained Models**: You can download pretrained models by running:

    ```bash
    bash download_models.sh
    ```

    For each task (IMDB/Newsgroups/RACE), this script saves the:

    - Predictor model to: `trained_predictors/{TASK}/model/`.
    - Editor checkpoint to: `results/{TASK}/editors/{EXPERIMENT-NAME}/{TASK}_editor.pth`.

3. **Generate Edits**: Run the following command to generate edits for a particular task with our pretrained editor. It will write edits to `results/{TASK}/edits/{STAGE2EXP}/edits.csv`.

    ```bash
    python run_stage_two.py -task {TASK} -stage2_exp {STAGE2EXP} -editor_path results/{TASK}/editors/mice/{TASK}_editor.pth
    ```

    For instance, to generate edits for the IMDB task, the following command will save edits to `results/imdb/edits/mice_binary/edits.csv`:

    ```bash
    python run_stage_two.py -task imdb -stage2_exp mice_binary -editor_path results/imdb/editors/mice/imdb_editor.pth
    ```

4. **Inspect Edits**: Inspect these edits with the demo notebook `notebooks/evaluation.ipynb`.

## More Information

### Training Editors

The following command will train an editor (i.e. run Stage 1 of MiCE) for a particular task. It saves checkpoints to `results/{TASK}/editors/{STAGE1EXP}/checkpoints/`.

´python run_stage_one.py -task {TASK} -stage1_exp {STAGE1EXP}´

### Generating Edits

The following command will find MiCE edits (i.e. run Stage 2 of MiCE) for a particular task. It saves edits to `results/{TASK}/edits/{STAGE2EXP}/edits.csv`. `-editor_path` determines the Editor model to use. Defaults to our pretrained Editor.

´python run_stage_two.py -task {TASK} -stage2_exp {STAGE2EXP} -editor_path results/{TASK}/editors/mice/{TASK}_editor.pth´

### Inspecting Edits

  The notebook `notebooks/evaluation.ipynb` contains some code to inspect edits.
  To compute fluency of edits, see the `EditEvaluator` class in `src/edit_finder.py`.

## Adding a Task

Follow the steps below to extend this repo for your own task.

1. Create a subfolder within `src/predictors/{TASK}`

2. **Dataset reader**: Create a task specific dataset reader in a file `{TASK}_dataset_reader.py` within that subfolder. It should have methods: `text_to_instance()`, `_read()`, and `get_inputs()`.

3. **Train Predictor**: Create a training config (see `src/predictors/imdb/imdb_roberta.json` for an example). Then train the Predictor using AllenNLP (see above commands or commands in `run_all.sh` for examples).

4. **Train Editor Model**: Depending on the task, you may have to create a new `StageOneDataset` subclass (see `RaceStageOneDataset` in `src/dataset.py` for an example of how to inherit from `StageOneDataset`).
    - For classification tasks, the existing base `StageOneDataset` class should work.
    - For new multiple-choice QA tasks with dataset readers patterned after the `RaceDatasetReader` (`src/predictors/race/race_dataset_reader.py`), the existing `RaceStageOneDataset` class should work.

5. **Generate Edits**: Depending on the task, you may have to create a new `Editor` subclass (see `RaceEditor` in `src/editor.py` for an example of how to inherit from `Editor`).
    - For classification tasks, the existing base `Editor` class should work.
    - For multiple-choice QA with dataset readers patterned after `RaceDatasetReader`, the existing `RaceEditor` class should work.

## Citation

If you want to cite the original paper please give them a cite!

```bibtex
@inproceedings{Ross2020ExplainingNM,
    title = "Explaining NLP Models via Minimal Contrastive Editing (MiCE)",
    author = "Ross, Alexis  and Marasovi{\'c}, Ana  and Peters, Matthew E.",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    publisher = "Association for Computational Linguistics",
    url= "https://arxiv.org/abs/2012.13985",
}
```
