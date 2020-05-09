from sklearn.metrics import roc_auc_score, log_loss


def cal_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def cal_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)


# y_true = [1,0,1,0]
# y_pred = [0.9, 0.1, 0.8, 0.2]

# print(cal_auc(y_true, y_pred))
# print(cal_logloss(y_true, y_pred))
