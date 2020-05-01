# -*- coding: utf-8 -*-
from ajctr.helpers import log, pathify
from ajctr.data.movielense import load_movielens


def make_movielense():
    movielens = load_movielens()
    log.info('movielens shape: {}'.format(movielens.shape))
    movielens.to_csv(pathify('data', 'interim', 'movielens.csv'))


def make_dataset():
    make_movielense()
