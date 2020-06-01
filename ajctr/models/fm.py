import numpy as np
from fastfm import als
from scipy.sparse import csr_matrix
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle


@timing
def train_fm_model():
    model = als.fmregression(n_iter=100, l2_reg_w=0.1, l2_reg_v=0.1, rank=10, random_state=0)
    x_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    x_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

    x_train = csr_matrix(x_train, dtype=np.float)
    x_val = csr_matrix(x_val, dtype=np.float)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    save_pickle(model, pathify('models', 'avazu-fm.pickle'))
    return model
