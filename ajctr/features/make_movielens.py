# -*- coding: utf-8 -*-
import re
import pandas as pd
from ajctr.helpers import log, pathify, save_pickle


_GENRES = [
	'Action',
	'Adventure',
	'Animation',
	'Children\'s',
	'Comedy',
	'Crime',
	'Documentary',
	'Drama',
	'Fantasy',
	'Film-Noir',
	'Horror',
	'Musical',
	'Mystery',
	'Romance',
	'Sci-Fi',
	'Thriller',
	'War',
	'Western'
]


def encode_gender(gender):
    return 1 if gender == 'F' else 0


def make_click_from_rating(rating):
    return 1 if rating == 5 else 0


def make_genre_list_from_string(genre_string):
    genres = genre_string.split('|')
    return [1 if g in genres else 0 for g in _GENRES] 


def extract_genres(movielens):
    genres_after_expand = pd.DataFrame(
        movielens['Genres'].map(make_genre_list_from_string).tolist(),
        columns=_GENRES
    )
    return pd.concat([movielens, genres_after_expand], axis=1, sort=False)


def extract_year_from_title(title):
    """
    Example:
        James and the Giant Peach (1996) -> 1996
        James and (1999) the Giant Peach (1996) -> 1999
        James and the Giant Peach -> 1900
    """
    match = re.compile(r'\(\d{4}\)').search(title)
    if match:
        return match.group(0)[1:-1]
    return 1900


def extract_debut_year(movielens):
    movielens['debut_year'] = movielens['Title'].map(extract_year_from_title)
    return movielens


def extract_features(movielens):
    movielens = extract_genres(movielens)
    movielens = extract_debut_year(movielens)
    return movielens


def make_binary_label(movielens):
    movielens['Click'] = movielens['Rating'].map(make_click_from_rating)
    return movielens


def preprocess_features(movielens):
    movielens['Gender'] = movielens['Gender'].map(encode_gender)
    return movielens


def make(is_debug=False):
    movielens = pd.read_csv(pathify('data', 'interim', 'movielens.csv'))
    movielens = extract_features(movielens)
    movielens = make_binary_label(movielens)
    movielens = preprocess_features(movielens)
    movielens.to_csv(
        pathify('data', 'interim', 'movielens-train-test.csv'),
        index=False
    )
