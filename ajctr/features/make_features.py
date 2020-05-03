# -*- coding: utf-8 -*-
from ajctr.helpers import log, pathify, save_pickle
from ajctr.features import make_movielens
from ajctr.features import make_avazu


def make_and_save_movielens():
    x, y = make_movielens.make()
    log.info('movielens X shape: {}'.format(x.shape))
    log.info('movielens y shape: {}'.format(y.shape))
    save_pickle(
        (x, y),
        pathify('data', 'processed', 'movielens.pickle')
    )

    
def make_and_save_avazu():
    x, y = make_avazu.make()
    log.info('avazu X shape: {}'.format(x.shape))
    log.info('avazu y shape: {}'.format(y.shape))
    save_pickle(
        (x, y),
        pathify('data', 'processed', 'avazu.pickle')
    )

    
def make():
    log.info('For movielens')
    make_and_save_movielens()
    log.info('For avazu')
    make_and_save_avazu()
