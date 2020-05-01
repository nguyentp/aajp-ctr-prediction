# -*- coding: utf-8 -*-
from ajctr.helpers import log, pathify
from ajctr.data import make_movielens


def make_and_save_movielens():
    movielens = make_movielens.make()
    log.info('movielens shape: {}'.format(movielens.shape))
    movielens.to_csv(
        pathify('data', 'interim', 'movielens.csv'), 
        index=False
    )


def make():
    make_and_save_movielens()
