from math import exp, log, sqrt
import numpy as np
from ajctr.reports.metrics import cal_auc, cal_logloss
from ajctr.helpers import load_processed_data, pathify, log, timing, save_pickle
from tqdm import tqdm

@timing
def train_ftlr_model():
    X_train, y_train = load_processed_data(pathify('data', 'processed', 'avazu-cv-train.csv'), label_col='click')
    X_val, y_val = load_processed_data(pathify('data', 'processed', 'avazu-cv-val.csv'), label_col='click')
    
    params = {
        'alpha':0.1,            # learning rate
        'beta':1,               # smoothing parameter for adaptive learning rate
        'L1':1,                 # L1 regularization, larger value means more regularized
        'L2':1,                 # L2 regularization, larger value means more regularized
        'num_categories':2**16, # make sure it is the same value with make_features.py
        'verbose':True
    }
    ftlr = ftlr_proximal(**params)
    ftlr.fit(X_train, y_train, X_val, y_val)

    y_pred = []
    for x_val in list(X_val.values):
        p = ftlr.predict(x_val)
        y_pred.append(p)
    y_pred = np.array(y_pred)
    auc_score = cal_auc(y_val, y_pred)
    log.info("auc_score: {:.4f}".format(auc_score))

    log_loss = cal_logloss(y_val, y_pred)
    log.info("log_loss: {:.4f}".format(log_loss))

    save_pickle(ftlr, pathify('models', 'avazu-ftlr.pickle'))
    return ftlr

class ftlr_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, num_categories, verbose):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * num_categories
        self.z = [0.] * num_categories
        self.w = {}

        self.verbose = verbose

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

    def fit(self, X_train, y_train, X_val, y_val):
        # convert dataframe to numpy array
        X_train = X_train.values
        y_train = y_train.values
        X_val = X_val.values
        y_val = y_val.values

        count = 1
        for x, y in tqdm(zip(X_train, y_train)):
            # if (count) % 10000 == 0:
            #     y_pred = []
            #     for x_val in list(X_val):
            #         p = self.predict(x_val)
            #         y_pred.append(p)
            #     y_pred = np.array(y_pred)
            #     log_loss = cal_logloss(y_val, y_pred)
            #     print("{} {:.5f}".format(count, log_loss))

            p = self.predict(x)
            self.update(x, p, y)
            count += 1

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g