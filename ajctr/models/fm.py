import numpy as np
from fastFM import mcmc
from scipy.sparse import csr_matrix
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle
from sklearn.preprocessing import OneHotEncoder


@timing
def train_fm_model():
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')
    
    encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)


    X_train = csr_matrix(X_train)
    X_val = csr_matrix(X_val)
    y_train[y_train == 0] = -1
    y_val[y_val == 0] = -1
    y_train = np.array(y_train)
    y_val   = np.array(y_val)

    fm = mcmc.FMClassification(n_iter=50, init_stdev=0.1, random_state= 123, rank=2)
    y_pred = fm.fit_predict_proba(X_train, y_train, X_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    save_pickle(fm, pathify('models', 'avazu-fm.pickle'))
    return fm

