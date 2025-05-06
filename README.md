# ESN_dual_channel

## Related Paper

This project is based on the paper:  
**[A Dynamic Systems Approach to Modeling Human–Machine Rhythm Interaction](https://ieeexplore.ieee.org/abstract/document/10938637)**  
Published in *IEEE Transactions on Cybernetics*.

## Dataset

The test dataset used for evaluation in the paper is provided in the file `rhythm_test2_new.mat`.

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
