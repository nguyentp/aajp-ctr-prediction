import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle


@timing
def train_logistic_model():
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')
    encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)
    # B/c all features after onehot is 0/1.

    params = {
        'penalty':'l2',
        'C':100.0,
        'class_weight':'balanced',
        'solver':'saga',
        'max_iter':500,
        'verbose':1,
        'n_jobs':-1
    }
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(**params))
    ])
    lr.fit(X_train, y_train)

    y_pred = lr.predict_proba(X_val)[:, 1]
    
    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    save_pickle(lr, pathify('models', 'avazu-lr.pickle'))
    return lr
