from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from fastFM import als, sgd, mcmc
from scipy.sparse import csr_matrix
import numpy as np
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle, load_pickle
from ajctr.models.fm import make_fm_model
import time
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def train_logistic_model():
    params = {
        'penalty': 'l2',
        'C': 1.0,
        'random_state': None,
        'solver': 'lbfgs',
        'max_iter': 300,
        'verbose': 0,
    }

    lr = LogisticRegression(**params)
    train_avazu(lr, model_name='lr', is_saving=True)

    params = {
        'penalty': 'l2',
        'C': 1.0,
        'random_state': None,
        'solver': 'lbfgs',
        'max_iter': 300,
        'verbose': 0,
    }

    lr = LogisticRegression(**params)
    train_movie_lens(lr, model_name='lr', is_saving=True)


def train_gradientboosting_model():
    params = {
        'learning_rate': 0.1,
        'colsample_bytree': 1,
        'n_estimators': 100,
        'gamma': 1,
        'max_depth': 3,
        'lambda': 1,
        'subsample': 0.8,
        'verbosity': 0,
    }

    gb = xgb.XGBClassifier(**params)
    train_avazu(gb, model_name='gb', is_saving=True)

    params = {
        'learning_rate': 0.1,
        'colsample_bytree': 1,
        'n_estimators': 100,
        'gamma': 1,
        'max_depth': 3,
        'lambda': 1,
        'subsample': 0.8,
        'verbosity': 0,
    }

    gb = xgb.XGBClassifier(**params)
    train_movie_lens(gb, model_name='gb', is_saving=True)


def train_fm_model():
    train_fastFM_avazu('fastFM', is_saving=True)
    # train_fastFM_movielens('fastFM', is_saving= True)
    # train_fm_avazu('FM', is_saving= True)
    # train_fm_movielens('FM', is_saving= True)


@timing
def train_avazu(model, model_name, is_saving=True):
    X_train, y_train = load_processed_data(
        pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(
        pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    if is_saving:
        save_pickle(model, pathify(
            'models', 'avazu-{}.pickle'.format(model_name)))
    return model


@timing
def train_movie_lens(model, model_name, is_saving=True):

    auc_scores = []
    log_losses = []
    for fold in range(5):
        log.info("Fold: {}".format(fold))
        X_train, y_train = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )
        X_val, y_val = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        auc_scores.append(cal_auc(y_val, y_pred))
        log_losses.append(cal_logloss(y_val, y_pred))

        if is_saving:
            save_pickle(model, pathify(
                'models', 'movielens-{}-cv{}.pickle'.format(model_name, fold)))

    log.info(
        "auc score: {:.4f}+-{:.4f}".format(np.mean(auc_scores), np.std(auc_scores)))

    log.info(
        "log_loss: {:.4f}+-{:.4f}".format(np.mean(log_losses), np.std(log_losses)))


@timing
def train_fastFM_movielens(model_name, is_saving=True):
    model = als.FMRegression(n_iter=100, l2_reg_w=0.1,
                             l2_reg_V=0.1, rank=10, random_state=0)
    auc_scores = []
    log_losses = []

    for fold in range(5):
        log.info("Fold: {}".format(fold))
        X_train, y_train = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )
        X_val, y_val = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )
        X_train = csr_matrix(X_train, dtype=np.float)
        X_val = csr_matrix(X_val, dtype=np.float)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        auc_scores.append(cal_auc(y_val, y_pred))
        log_losses.append(cal_logloss(y_val, y_pred))

        if is_saving:
            save_pickle(model, pathify(
                'models', 'movielens-{}-cv{}.pickle'.format(model_name, fold)))

    log.info(
        "auc score: {:.4f}+-{:.4f}".format(np.mean(auc_scores), np.std(auc_scores)))

    log.info(
        "log_loss: {:.4f}+-{:.4f}".format(np.mean(log_losses), np.std(log_losses)))

@timing
def train_fastFM_avazu(model_name, is_saving=True):
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')


    # nrows = 100000
    # label_col = 'click'
    # df = pd.read_csv(pathify('data', 'processed', 'avazu-cv-train.csv'), nrows=nrows, dtype=np.int16)
    # X_train = df.drop(columns=[label_col])
    # y_train = df[label_col]
    
    
    # df = pd.read_csv(pathify('data', 'processed', 'avazu-cv-train.csv'), dtype=np.int16)
    # X_val = df.drop(columns=[label_col])
    # y_val = df[label_col]
    

    # X_train = X_train[:5000000]
    # y_train = y_train[:5000000]
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # X_train = X_train % 1000
    # print(X_train.describe())

    
    encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    X_train = csr_matrix(X_train)
    X_val = csr_matrix(X_val)
    y_train[y_train == 0] = -1
    y_val[y_val == 0] = -1
    y_train = np.array(y_train)
    y_val   = np.array(y_val)

    start = time.time()
    model = mcmc.FMClassification(n_iter=50, init_stdev=0.1, random_state= 123, rank=2)
    y_pred = model.fit_predict_proba(X_train, y_train, X_val)
    auc_score_val = cal_auc(y_val, y_pred)
    log.info("auc_score_val: {:.4f}".format(auc_score_val))
    log.info("log_loss_val: {:.4f}".format(cal_logloss(y_val, y_pred)))
    print("time: {} min".format((time.time()-start)/60))
    if is_saving:
        save_pickle(model, pathify('models', 'avazu-{}.pickle'.format(model_name)))



@timing
def train_fm_movielens(model_name, is_saving=True):
    auc_scores = []
    log_losses = []

    for fold in range(5):
        log.info("Fold: {}".format(fold))
        X_train, y_train = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )
        X_val, y_val = load_processed_data(
            pathify('data', 'processed',
                    'movielens-cv{}-train.csv'.format(fold)),
            label_col='Click'
        )

        X_train.rename(columns={"Children's": 'Childrens'}, inplace=True)
        X_val.rename(columns={"Children's": 'Childrens'}, inplace=True)

        # treat all columns as categorical, temporary skip Children's
        num_names = []
        cat_names = list(X_train.columns)
        cat_nuniques = [X_train[f].max() for f in cat_names]
        fm = make_fm_model(num_names, cat_names, cat_nuniques, 3)
        # print(fm.summary())
        # plot_keras_model(fm, 'fm.png')
        fm.compile(loss='MSE', optimizer='adam')

        xtrain = [X_train[f].values.reshape(-1, 1)
                  for f in num_names + cat_names]
        xval = [X_val[f].values.reshape(-1, 1) for f in num_names + cat_names]
        ytrain = y_train.values.reshape(-1, 1)
        yval = y_val.values.reshape(-1, 1)

        fm.fit(xtrain, ytrain, batch_size=64,
               epochs=1, validation_data=(xval, yval))
        y_pred = fm.predict(xval, batch_size=64)

        auc_scores.append(cal_auc(y_val, y_pred))
        log_losses.append(cal_logloss(y_val, y_pred))

        if is_saving:
            save_pickle(fm, pathify(
                'models', 'movielens-{}-cv{}.pickle'.format(model_name, fold)))

    log.info(
        "auc score: {:.4f}+-{:.4f}".format(np.mean(auc_scores), np.std(auc_scores)))

    log.info(
        "log_loss: {:.4f}+-{:.4f}".format(np.mean(log_losses), np.std(log_losses)))


@timing
def train_fm_avazu(model_name, is_saving=True):
    X_train, y_train = load_processed_data(
        pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(
        pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

    # treat all columns as categorical
    num_names = []
    cat_names = list(X_train.columns)
    cat_nuniques = [X_train[f].max() for f in cat_names]
    fm = make_fm_model(num_names, cat_names, cat_nuniques, 3)
    # print(fm.summary())
    # plot_keras_model(fm, 'fm.png')
    fm.compile(loss='MSE', optimizer='adam')

    xtrain = [X_train[f].values.reshape(-1, 1) for f in num_names + cat_names]
    xval = [X_val[f].values.reshape(-1, 1) for f in num_names + cat_names]
    ytrain = y_train.values.reshape(-1, 1)
    yval = y_val.values.reshape(-1, 1)

    fm.fit(xtrain, ytrain, batch_size=64,
           epochs=3, validation_data=(xval, yval))
    y_pred = fm.predict(xval, batch_size=64)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    if is_saving:
        save_pickle(fm, pathify(
            'models', 'avazu-{}.pickle'.format(model_name)))


def train():
    # train_logistic_model()
    # train_gradientboosting_model()
    train_fm_model()
