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
