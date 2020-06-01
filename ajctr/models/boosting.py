import xgboost as xgb
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle


@timing
def train_gradientboosting_model():
    x_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    x_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')

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
    gb.fit(x_train, y_train, eval_metric='auc', verbose=True, eval_set=[(x_val, y_val)])
    y_pred = gb.predict_proba(x_val)[:, 1]
    
    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    save_pickle(gb, pathify('models', 'avazu-gb.pickle'))
    return gb
