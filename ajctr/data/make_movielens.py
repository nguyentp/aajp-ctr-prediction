# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ajctr.helpers import log, pathify


_SEP = '::'
_ENCODING = 'ISO-8859-1'
_PARSE_ENGINE = 'python'


def load_dat_file(path, headers):
    return pd.read_csv(
        path,
        sep=_SEP,
        names=headers,
        encoding=_ENCODING,
        engine=_PARSE_ENGINE
    )


def load_ratings():
    path = pathify('data', 'raw', 'ml-1m', 'ratings.dat')
    headers = 'UserID::MovieID::Rating::Timestamp'.split(_SEP)
    return load_dat_file(path, headers)


def load_movies():
    path = pathify('data', 'raw', 'ml-1m', 'movies.dat')
    headers = 'MovieID::Title::Genres'.split(_SEP)
    return load_dat_file(path, headers)


def load_users():
    path = pathify('data', 'raw', 'ml-1m', 'users.dat')
    headers = 'UserID::Gender::Age::Occupation::Zip-code'.split(_SEP)
    return load_dat_file(path, headers)


def make():
    ratings = load_ratings()
    log.info('ratings shape: {}'.format(ratings.shape))

    movies = load_movies()
    log.info('movies shape: {}'.format(movies.shape))

    users = load_users()
    log.info('users shape: {}'.format(users.shape))

    movielens = (
        ratings
        .merge(movies, on=['MovieID'])
        .merge(users, on=['UserID'])
    )
    movielens.to_csv(pathify('data', 'interim', 'movielens.csv'), index=False)
    log.info('Movielens after merge: {}'.format(movielens.shape))