# Text Classification Lab

This project is a text classification experiment based on the 20 Newsgroups dataset. It includes data preprocessing, feature extraction, model training, prediction scripts, and hyperparameter analysis for several classic machine learning classifiers.

## Project Structure

```text
text-classification-lab/
├── data/
│   ├── 20news-bydate-train/      # Original training dataset
│   ├── 20news-bydate-test/       # Original test dataset
│   ├── cleaned_train/            # Cleaned training dataset
│   └── cleaned_test/             # Cleaned test dataset
├── notebooks/
│   └── hyperparameter_analysis/  # Model hyperparameter analysis scripts
├── src/
│   ├── models/                   # KNN, Naive Bayes, and Perceptron models
│   ├── data_explore.py           # Dataset exploration
│   ├── feature_extraction.py     # Vocabulary and sparse representation generation
│   ├── predict.py                # Prediction entry point
│   ├── text_cleaning.py          # Text preprocessing
│   └── train.py                  # Model training entry point
├── sparse-representations.txt    # Sparse document representations
└── vocab-dictionary.json         # Vocabulary dictionary
```

## Dataset

The `data/` directory contains the 20 Newsgroups dataset and cleaned versions used for training and testing. The dataset is organized by category folders, so each folder name can be used as the label for the documents inside it.

## Models

The project currently includes:

- Perceptron
- K-Nearest Neighbors
- Naive Bayes

Each model implementation is stored in `src/models/`.

## Usage

Install the required Python packages first:

```powershell
pip install scikit-learn scipy numpy joblib
```

Train a model:

```powershell
python src/train.py --model perceptron
python src/train.py --model knn
python src/train.py --model naive_bayes
```

Optional arguments:

```powershell
python src/train.py --model perceptron --train_data data/cleaned_train --test_data data/cleaned_test --output_dir outputs
```

Run prediction after a trained model has been saved:

```powershell
python src/predict.py
```

## Feature Extraction

`src/feature_extraction.py` generates:

- `vocab-dictionary.json`: a vocabulary dictionary
- `sparse-representations.txt`: sparse document representations

These files can be used to inspect the generated text features.

## Notes

- Generated model files and experiment outputs are saved under `outputs/`, which is ignored by Git.
- IDE settings, Python cache files, virtual environments, and log files are also ignored.
- The dataset is included in this repository because the files are small enough for a normal GitHub repository.
