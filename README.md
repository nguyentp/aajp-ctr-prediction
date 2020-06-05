# Introduction

> Predict Click-Through-Rate with different Machine Learning models.

In this project, we will provide a benchmark of different Algorithms on Click-Through-Rate (CTR) dataset.

This project can be useful in 2 ways:

1. ***Quick*** check on common technique used in CTR prediction. If you have a dataset similar to CTR Dataset, and you want to get a baseline score, you can try to run this project on new dataset. 
2. Used as base project and ***can be extend*** with more advanced algorithms like (Deep Learning model) and various CTR datasets. As the first version, we will use only 1 dataset and 4 basic algoithms.

# Datasets

Some common properties of CTR dataset are:

- Number of rows is large: several millions or more.
- Almost features are Categorical Features.
- Categorical features are High-Cardinaliry: Number of unique values are large (normaly, thousands values).
- Target is highly imbalance.

For each dataset, we report some summary statistics to illustrate the characteristics of CTR dataset. If your dataset has similar characteristics, you can try techniques in this project as well.

## [Avazu Click-Through Rate](https://www.kaggle.com/c/avazu-ctr-prediction/data)

- Dataset contains Online Ads informations and its label (Click or No Click) in 10 days. Algorithms have to predict Ads Click probability. Data is ordered chronologically. Data can be downloaded from Kaggle. In this project, we will use only Train data to benchmark algorithm.

  - Total rows: 40 millions (40,428,967 rows).
  - Date ranges: 10 days, from 2014/10/21 to 2014/10/30.
  - Average number of rows-per-day: 4 millions (4,042,896/day).
  - Click Rate on data: 0.1698

- Each rows of data has these information:
  - `id`: Ads identifier
  - `click`: 0/1 for non-click/click
  - `hour`: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
  - `banner_pos`
  - `site_id`
  - `site_domain`
  - `site_category`
  - `app_id`
  - `app_domain` 
  - `app_category` 
  - `device_id`
  - `device_ip`
  - `device_model` 
  - `device_type`
  - `device_conn_type`
  - `C1`: anonymized categorical variable
  - `C14-C21`: anonymized categorical variables

- Sample rows:

|    |                   id |   click |     hour |   C1 |   banner_pos | site_id   | site_domain   | site_category   | app_id   | app_domain   | app_category   | device_id   | device_ip   | device_model   |   device_type |   device_conn_type |   C14 |   C15 |   C16 |   C17 |   C18 |   C19 |    C20 |   C21 |
|---:|---------------------:|--------:|---------:|-----:|-------------:|:----------|:--------------|:----------------|:---------|:-------------|:---------------|:------------|:------------|:---------------|--------------:|-------------------:|------:|------:|------:|------:|------:|------:|-------:|------:|
|  0 |  1000009418151094273 |       0 | 14102100 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | ddd2926e    | 44956a24       |             1 |                  2 | 15706 |   320 |    50 |  1722 |     0 |    35 |     -1 |    79 |
|  1 | 10000169349117863715 |       0 | 14102100 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | 96809ac8    | 711ee120       |             1 |                  0 | 15704 |   320 |    50 |  1722 |     0 |    35 | 100084 |    79 |
|  2 | 10000371904215119486 |       0 | 14102100 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | b3cf8def    | 8a4875bd       |             1 |                  0 | 15704 |   320 |    50 |  1722 |     0 |    35 | 100084 |    79 |

- Summary staticstics:

|    | Column           |   #Unique |
|---:|:-----------------|----------:|
|  0 | C1               |         7 |
|  1 | banner_pos       |         7 |
|  2 | site_id          |      4737 |
|  3 | site_domain      |      7745 |
|  4 | site_category    |        26 |
|  5 | app_id           |      8552 |
|  6 | app_domain       |       559 |
|  7 | app_category     |        36 |
|  8 | device_id        |   2686408 |
|  9 | device_ip        |   6729486 |
| 10 | device_model     |      8251 |
| 11 | device_type      |         5 |
| 12 | device_conn_type |         4 |
| 13 | C14              |      2626 |
| 14 | C15              |         8 |
| 15 | C16              |         9 |
| 16 | C17              |       435 |
| 17 | C18              |         4 |
| 18 | C19              |        68 |
| 19 | C20              |       172 |
| 20 | C21              |        60 |


# Cross Validation

## Avazu Click-Through Rate

- We use data from date `2014/10/21 - 2014/10/29` as training data. And use last date `2014/10/30` for Validation.
- Metric used to report Validation score are: AUC and LogLoss.

# Data Processing and Feature Engineering

## Avazu Click-Through Rate

- Feature Extraction:
  - Extract hour in day from `hour` column. For example: `14102100 -> 00`
  - Extract count features for each of below columns. For example: How many time we have seen `device_id == a99f214a` in data? Use only training data to estimate count features. In validation data, map count features for corresponding column.
    - `device_id`
    - `device_ip`
    - `device_id + device_ip`
    - `hour`
  - Sample data after feature extraction:

|    |   click |                   id |   hour |   C1 |   banner_pos | site_id   | site_domain   | site_category   | app_id   | app_domain   | app_category   | device_id   | device_ip   | device_model   |   device_type |   device_conn_type |   C14 |   C15 |   C16 |   C17 |   C18 |   C19 |    C20 |   C21 |   device_id_count |   device_ip_count |   user_id_count |   hour_count |
|---:|--------:|---------------------:|-------:|-----:|-------------:|:----------|:--------------|:----------------|:---------|:-------------|:---------------|:------------|:------------|:---------------|--------------:|-------------------:|------:|------:|------:|------:|------:|------:|-------:|------:|------------------:|------------------:|----------------:|-------------:|
|  0 |       0 |  1000009418151094273 |      0 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | ddd2926e    | 44956a24       |             1 |                  2 | 15706 |   320 |    50 |  1722 |     0 |    35 |     -1 |    79 |               869 |                 5 |               1 |          999 |
|  1 |       0 | 10000169349117863715 |      0 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | 96809ac8    | 711ee120       |             1 |                  0 | 15704 |   320 |    50 |  1722 |     0 |    35 | 100084 |    79 |               869 |                 1 |               1 |          999 |
|  2 |       0 | 10000371904215119486 |      0 | 1005 |            0 | 1fbe01fe  | f3845767      | 28905ebd        | ecad2386 | 7801e8d9     | 07d7df22       | a99f214a    | b3cf8def    | 8a4875bd       |             1 |                  0 | 15704 |   320 |    50 |  1722 |     0 |    35 | 100084 |    79 |               869 |                 1 |               1 |          999 |


- Preprocessing: A common practice in preprocessing CTR data is Hash Trick. For each feature, hash feature's value into fixed size vector. In current processing step, we hash value into `2^16` fixed size vector. If data has `n` features, after preprocessing, we expand input dimension space into a larger output dimension space `n * 2^16`. For space efficiency, we only store index of feature's value. For example: Site ID value after hashing has index = 123, we store: `site_id: 123`.
  - Logistic Regression and Factorization Machine: Before train and predict, we one-hot-encode input feature.
  - FTRL and GBM: Do nothing.
  - Example data after preprocessing: 

|    |   click |    C1 |   banner_pos |   site_id |   site_domain |   site_category |   app_id |   app_domain |   app_category |   device_id |   device_ip |   device_model |   device_type |   device_conn_type |   C14 |   C15 |   C16 |   C17 |   C18 |   C19 |   C20 |   C21 |   device_id_count |   device_ip_count |   user_id_count |   hour_count |
|---:|--------:|------:|-------------:|----------:|--------------:|----------------:|---------:|-------------:|---------------:|------------:|------------:|---------------:|--------------:|-------------------:|------:|------:|------:|------:|------:|------:|------:|------:|------------------:|------------------:|----------------:|-------------:|
|  0 |       0 | 49542 |        38901 |      3703 |         60146 |           37267 |     4859 |          931 |          52646 |        2603 |       57603 |          65226 |         16939 |              52576 | 59061 | 26495 | 29876 | 35552 | 50048 | 42447 | 51572 | 26673 |             64040 |             26097 |           14350 |         8625 |
|  1 |       0 | 49542 |        38901 |      3703 |         60146 |           37267 |     4859 |          931 |          52646 |        2603 |       25930 |          60692 |         16939 |              15778 | 35267 | 26495 | 29876 | 35552 | 50048 | 42447 | 15865 | 26673 |             64040 |             61530 |           14350 |         8625 |
|  2 |       0 | 49542 |        38901 |      3703 |         60146 |           37267 |     4859 |          931 |          52646 |        2603 |       10179 |          56745 |         16939 |              15778 | 35267 | 26495 | 29876 | 35552 | 50048 | 42447 | 15865 | 26673 |             64040 |             61530 |           14350 |         8625 |

# Models

Below are models we use in project. (This is just tentative list. We can add or remove some of them later).

1. Logistic Regression (LR): A linear model, good for strong baseline.
2. Gradient Boosting Machine (GBM): A non-linear model, works on various dataset. We want to check how Boosting works on this data. It is also a strong baseline too.
3. Matrix Factorization (FM): A go-to algorithm for CTR problem. Detail of algorithm, see below.
4. FTRL-Proximal online learning algorithm (FTRL): A research from Google for CTR prediction. It bases on Logistic Regression but support some useful features: Online-Learning and low memory footprint. Detail of algorithm, see below.

## Factorization Machine:

## FTLR:

# Results

to be updating...

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
    │
    ├── ajctr              <- Source code for use in this project.
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
    │   └── reports  <- Scripts to create exploratory and results oriented visualizations
    │       └── metrics.py
    ├── main.py            <- Program Entry point
    │
    ├── test               <- Test code

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