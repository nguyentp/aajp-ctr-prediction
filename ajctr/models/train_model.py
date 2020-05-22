from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle

def train_logistic_model():
    params = {
        'penalty':'l2',
        'C':1.0,
        'random_state':None,
        'solver':'lbfgs',
        'max_iter':300,
        'verbose':0,
    }

    lr = LogisticRegression(**params)
    train_avazu(lr, model_name='lr', is_saving=True)

    params = {
        'penalty':'l2',
        'C':1.0,
        'random_state':None,
        'solver':'lbfgs',
        'max_iter':300,
        'verbose':0,
    }

    lr = LogisticRegression(**params)
    train_movie_lens(lr, model_name='lr', is_saving=True)


def train_gradientboosting_model():
    params = {
        'learning_rate':0.1,
        'colsample_bytree':1,
        'n_estimators':100,
        'gamma':1,
        'max_depth':3,
        'lambda':1,
        'subsample':0.8,
        'verbosity':0,
    }

    gb = xgb.XGBClassifier(**params)
    train_avazu(gb, model_name='gb', is_saving=True)

    params = {
        'learning_rate':0.1,
        'colsample_bytree':1,
        'n_estimators':100,
        'gamma':1,
        'max_depth':3,
        'lambda':1,
        'subsample':0.8,
        'verbosity':0,
    }

    gb = xgb.XGBClassifier(**params)
    train_movie_lens(gb, model_name='gb', is_saving=True)
    

@timing
def train_avazu(model, model_name, is_saving=True):
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    if is_saving:
            save_pickle(model, pathify('models', 'avazu-{}.pickle'.format(model_name)))
    return model

@timing
def train_movie_lens(model, model_name, is_saving=True):

    auc_scores = []
    log_losses = []
    for fold in range(5):
        log.info("Fold: {}".format(fold))
        X_train, y_train = load_processed_data(
            pathify('data', 'processed', 'movielens-cv{}-train.csv'.format(fold)), 
            label_col='Click'
        )
        X_val, y_val = load_processed_data(
            pathify('data', 'processed', 'movielens-cv{}-train.csv'.format(fold)), 
            label_col='Click'
        )


        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        auc_scores.append(cal_auc(y_val, y_pred))
        log_losses.append(cal_logloss(y_val, y_pred))

        if is_saving:
            save_pickle(model, pathify('models', 'movielens-{}-cv{}.pickle'.format(model_name, fold)))

    log.info("auc score: {:.4f}+-{:.4f}".format(np.mean(auc_scores), np.std(auc_scores)))

    log.info("log_loss: {:.4f}+-{:.4f}".format(np.mean(log_losses), np.std(log_losses)))

def train():
    # train_logistic_model()
    train_gradientboosting_model()
