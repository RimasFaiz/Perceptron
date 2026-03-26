# Binary Perceptron Classifier (perceptron.py)

A small, self-contained binary perceptron implementation for experimenting with pairwise binary classification (originally written for the Iris-like dataset). The script trains three binary perceptrons (class-1 vs class-2, class-2 vs class-3, class-1 vs class-3), reports train/test accuracy for each pair, and prints the most discriminative feature according to the learned weights.

---

## Features

- Simple perceptron learning algorithm (online updates).
- Handles bias term internally.
- Filters dataset into binary problems for pairwise classifiers.
- Prints training and testing accuracy, and the most discriminative feature (by absolute weight).
- Lightweight: uses only NumPy and pandas.

---

## Requirements

- Python 3.7+
- numpy
- pandas

Install dependencies with pip:

pip install numpy pandas

---

## Data format / Expectations

- The script expects CSV files with no header and five columns in this order:
  1. f1 (float)
  2. f2 (float)
  3. f3 (float)
  4. f4 (float)
  5. label (string)

- Labels must match the class names used in `class_pairs` inside the script. By default the script uses:
  - `class-1`, `class-2`, `class-3`

- Example CSV row (single-line):
  5.1,3.5,1.4,0.2,class-1

- The code maps the first class in each pair to label `+1` and the second class to `-1`.

---

## Usage

By default the script loads these files:

- `attached_assets/train.data`
- `attached_assets/test.data`

Run the script:

python perceptron.py

Output will include:
- Train and test accuracy (percentage) for each pair.
- The most discriminative feature and its learned weight.
- A summary indicating which pair was the most difficult to separate (lowest test accuracy).

If your files are located elsewhere, either modify the `train_file` and `test_file` variables in `main()` or adapt the script to accept command-line arguments (suggestion below).

---

## Quick customization suggestions

- Accept command-line arguments (recommended): use `argparse` to pass paths for training/testing files, number of iterations, and class labels.
- Use scaling/normalization: Perceptron performance can improve if features are standardized (e.g., using scikit-learn's `StandardScaler`).
- Shuffle training data each epoch to reduce ordering bias.
- Add a learning rate (currently the perceptron uses a learning rate of 1).
- Save trained weights to a file for later inspection or reuse.

Example change to add argparse (short suggestion):
- Allow `--train`, `--test`, `--iterations` and `--classes` flags, and validate files exist.

---

## Understanding the code (brief)

- `load_data(filename)` â reads the CSV (no header) into NumPy arrays: features and labels.
- `filter_data(X, y, class1, class2)` â selects rows for the two classes and converts labels to `+1` / `-1`.
- `Perceptron` class:
  - `__init__(num_features, iterations=20)` â initializes weights (including bias) to zero and sets the number of iterations (epochs).
  - `train(X, y)` â trains the perceptron using the standard update rule when `y * (w Â· x) <= 0`.
  - `predict(X)` â returns predictions in {+1, -1}.
  - `accuracy(X, y)` â returns accuracy in percent.
- `get_most_discriminative_feature(weights)` â returns the feature with the largest absolute weight (bias excluded); the script labels features with Iris-style names by default.
- `main()` â orchestrates loading data, training three pairwise classifiers, printing results, and summarizing the most difficult pair.

---

## Output interpretation

- Train Accuracy vs Test Accuracy: show if the perceptron generalizes well. Large train accuracy but low test accuracy may indicate overfitting or a non-representative train set.
- Most Discriminative Feature: feature with largest absolute weight; a larger magnitude indicates a stronger linear contribution in the decision hyperplane for that binary classifier.
- Most Difficult Pair: the pair with the lowest test accuracy â indicates classes that are least linearly separable with the current features.
