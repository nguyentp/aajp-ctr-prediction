from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_pickle, pathify, log, timing

def train_logistic_model():
    params = {
        'penalty':'l2',
        'C':1.0,
        'random_state':None,
        'solver':'lbfgs',
        'max_iter':100,
        'verbose':0,
    }
    lr = LogisticRegression(**params)

    train_avazu(lr)
    train_movie_lens(lr)


def train_gradientboosting_model():
    params = {
        'learning_rate':0.3,
        'gamma':0,
        'max_depth':6,
        'min_child_weight':1,
        'lambda':1,
        'alpha':0,
        'verbosity':0,
    }

    gb = xgb.XGBClassifier(**params)
    train_avazu(gb)
    train_movie_lens(gb)
    

@timing
def train_avazu(model):
    X_train, X_val, y_train, y_val = train_val_split_avazu()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {}".format(log_loss))

    return model

@timing
def train_movie_lens(model):
    # dummy data
    import numpy as np
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1]])
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0])

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    fold = 1
    for train_index, val_index in skf.split(X, y):
        log.info("Fold: {}".format(fold))
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        auc_score = cal_auc(y_val, y_pred)
        log.info("auc_score: {}".format(auc_score))

        log_loss = cal_logloss(y_val, y_pred)
        log.info("log_loss: {}".format(log_loss))

        fold += 1

#TODO: split the last day data as val set, remainings as train set
def train_val_split_avazu():
    # return dummy value for testing
    import numpy as np
    X_train = np.array([[1, 0, 1], [1, 1, 0]])
    X_val = np.array([[1, 1, 1], [0, 0, 0]])
    y_train = np.array([1, 0])
    y_val = np.array([1, 0])
    return X_train, X_val, y_train, y_val 

def train():
    train_logistic_model()
    train_gradientboosting_model()
