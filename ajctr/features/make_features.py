# -*- coding: utf-8 -*-
from ajctr.helpers import log, pathify, save_pickle
from ajctr.features import make_movielens


def make_and_save_movielens():
    x, y = make_movielens.make()
    log.info('movielens X shape: {}'.format(x.shape))
    log.info('movielens y shape: {}'.format(y.shape))
    save_pickle(
        (x, y),
        pathify('data', 'processed', 'movielens.pickle')
    )


def make():
    make_and_save_movielens()
