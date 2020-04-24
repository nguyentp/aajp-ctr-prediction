# Introduction

> Predict Click with different Machine Learning models on several Datasets.

We will submit this project to FPT SKU repository. To do that, we need below requirements:

## Quality Requirements

1. Try to develop project in Python and Tensorflow (TF) with Keras API.
   1. Why TF? It is popular and easy to use. TF supports production better than other frameworks.
   2. Swich to other framework when there is no efficient implementation in Python and Tensorflow.
2. Teams in FWI AAA can reference our project for ideas, results and code samples.
3. Source code is runable, readable and we can extend it or re-write it into other frameworks easily.
4. Document is clear and well-organized.

## Technical Requirements

- Report Datasets and how did we preprocessing data.
- Report Evaluation method and Ranking Metrics.
- Report Training time and Inference time on batch of data.
- Report Model size.
- Report Pros and Cons of each model.

# Models

Below are models we use in project, sorted by priority. This is just tentative list. We can add or remove some of them later.

1. Logistic Regression (LR).
2. Gradient Boosting Machine (GBM).
3. Singular Value Decomposition (SVD).
4. Matrix Factorization (FM).
5. FTRL-Proximal online learning algorithm (FTRL).
6. Neural Collaborative Filtering (NCF).
7. Wide & Deep Learning Model (WideDeep).
8. Bayesian Personalized Ranking (BPR).

# Datasets and Preprocessing

## Datasets

List of dataset we will use. We can add more later if time allow.

1. [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/data).
2. [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/).

## Data Processing


# Project Structure

```

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- All documents which is not code.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention: <id>-<author>-<desc>
    │                         - id: has 2 digits.
    │                         - author: name of developer.
    │                         - desc: a short description of notebook.
    │                         - Use hyphen "-" as separator.
    │                         - For example: `08-nguyentp2-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

```

# How to run entire project

1. Setup Environment with Anaconda and Python 3.6.
2. Download Datasets

```
chmod 755 download-data.sh
./download-data.sh
```

3. Install project in Developement mode:

```
pip install -e .
```

4. Run main program. Result will be in reports.

```
python ./main.py
```

# Project Developer must know

- Coding Convention:
  - We will use [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
  - This guide is long. If you are short of time, just read the `Decision` section. Or just read the interesting part.
- Project structure follow [Cookiecutter Data Science template](https://drivendata.github.io/cookiecutter-data-science/).
- Use Jupyter Notebook wisely! Notebook are good for Exploration and Communication.

## Some key naming conventions:

- Format source code with UTF-8.
- Use single quote `'` for string and double quotes `"` for docstring.
- Use 4 spaces indentions.
- Use underscore `_` and lowercase to name function, variable, modules. CamelCase for Class.

# Reference

- [Microsoft Recommenders repo](https://github.com/microsoft/recommenders).