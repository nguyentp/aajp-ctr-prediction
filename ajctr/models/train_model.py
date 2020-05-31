import numpy as np
import xgboost as xgb
from fastFM import als
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle
from ajctr.models.fm import make_fm_model


@timing
def train_logistic_model():
    params = {
        'penalty':'l2',
        'C':100.0,
        'class_weight':'balanced',
        'solver':'saga',
        'max_iter':500,
        'verbose':1,
        'n_jobs':-1
    }

    lr = LogisticRegression(**params)
    train_avazu(lr, model_name='lr', is_saving=True)


@timing
def train_gradientboosting_model():
    params = {
        'learning_rate':0.1,
        'colsample_bytree':0.8,
        'n_estimators':100,
        'gamma':1,
        'max_depth':6,
        'lambda':1,
        'min_child_weight':5
    }

    gb = xgb.XGBClassifier(**params)
    train_avazu(gb, model_name='gb', is_saving=True)


@timing
def train_fm_model():
    train_fastFM_avazu('fastFM', is_saving= True)


def train_avazu(model, model_name, is_saving=True):
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')
    if model_name == 'gb':
        model.fit(X_train, y_train, eval_metric='auc', verbose=True, eval_set=[(X_val, y_val)])
    elif model_name == 'lr':
        model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)
    
    auc_score = cal_auc(y_val, y_pred[:, 1])
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    if is_saving:
        save_pickle(model, pathify('models', 'avazu-{}.pickle'.format(model_name)))
    return model


def train_fastFM_avazu(model_name, is_saving= True):
    model = als.FMRegression(n_iter=100, l2_reg_w=0.1, l2_reg_V=0.1, rank=10, random_state=0)
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

    X_train = csr_matrix(X_train, dtype=np.float)
    X_val = csr_matrix(X_val, dtype=np.float)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    if is_saving:
        save_pickle(model, pathify('models', 'avazu-{}.pickle'.format(model_name)))
    return model


def train_fm_avazu(model_name, is_saving= True):
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')
    
    # treat all columns as categorical  
    num_names = []
    cat_names = list(X_train.columns)
    cat_nuniques = [X_train[f].max() for f in cat_names]
    fm = make_fm_model(num_names, cat_names, cat_nuniques, 3)
    # print(fm.summary())
    # plot_keras_model(fm, 'fm.png')
    fm.compile(loss = 'MSE', optimizer='adam')

    xtrain = [X_train[f].values.reshape(-1, 1) for f in num_names + cat_names]
    xval = [X_val[f].values.reshape(-1, 1) for f in num_names + cat_names]
    ytrain = y_train.values.reshape(-1, 1)
    yval = y_val.values.reshape(-1, 1)
  

    fm.fit(xtrain, ytrain, batch_size=64, epochs=3, validation_data=(xval, yval))
    y_pred = fm.predict(xval, batch_size=64)
    
    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))
   
    if is_saving:
        save_pickle(fm, pathify('models', 'avazu-{}.pickle'.format(model_name)))


def train():
    train_logistic_model()
    train_gradientboosting_model()
    train_fm_model()
