# ESN_dual_channel

## Related Paper

This project is based on the paper:  
**[A Dynamic Systems Approach to Modeling Human-Machine Rhythm Interaction](https://ieeexplore.ieee.org/abstract/document/10938637)**  
Published in *IEEE Transactions on Cybernetics*.

## Dataset

The original test dataset used for evaluation in the paper is provided in the file `rhythm_test2_new.mat`.

Additional data files used by the current dual-channel training/evaluation scripts are provided under `data/`:

- `data/rhythm_train4.mat`: MATLAB-generated two-channel rhythm training inputs.
- `data/rhythm_target4.mat`: training targets paired with `rhythm_train4.mat`.
- `data/rhythm_test2_new.mat`: test set loaded by the current `TimeSeriesDataset` evaluation path.

## Checkpoint

The main checkpoint used by `timeserie_prediction/human_prediction.py` is:

```text
checkpoint/saved_checkpint/final_version_4_first_jp.pt
```

Large `.mat` and `.pt` files are tracked with Git LFS.

## Figure Scripts

The plotting scripts used for the paper figures are collected in `paper_figure_scripts/`.
The scripts were renamed to describe the corresponding figure content more clearly:

- `fig_dual_channel_prediction_and_continuation_error.py`: dual-channel prediction and continuation error plots.
- `fig_dual_channel_update_and_continuation_offset.py`: dual-channel update and continuation time-offset plots.
- `fig_model_vs_human_ibi_scatter.py`: model-vs-human inter-beat interval scatter plots.
- `fig_human_computer_phase_frequency_distribution.py`: human-computer phase, frequency, and distribution comparison plots.
- `fig_human_human_phase_frequency_distribution.py`: human-human phase, frequency, and distribution comparison plots.
- `fig_model_comparison_and_adaptation_bars.py`: model comparison and adaptation bar plots.
- `fig_perturbation_conditions.py`: perturbation-condition plots for skipped and shifted beats.
- `fig_relative_phase_circle_plot.py`: relative-phase circle plots.

## Installation

To set up the environment:

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To reproduce the evaluation results:

1. Ensure the model code is located in the `echotorch/nn/` directory.
2. Open `run.py` and set the training state to `"test"`.
3. Run the script:

   ```bash
   python run.py
   ```

This will evaluate the model using the provided test dataset and replicate the results from the paper.

To regenerate the paper figures, run the corresponding script from `paper_figure_scripts/` after checking the data and output paths near the top of that script.
