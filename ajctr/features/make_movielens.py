# -*- coding: utf-8 -*-
import re
import csv
import pandas as pd
from sklearn.model_selection import GroupKFold
from ajctr.helpers import log, pathify, save_pickle, timing, iter_as_dict, categorize_by_hash


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


@timing
def split_for_validation(movielens):
    userids = movielens['UserID'].tolist()

    kfold = GroupKFold(n_splits=5)
    for i, (train_ids, val_ids) in enumerate(kfold.split(movielens, groups=userids)):
        train = movielens.iloc[train_ids, :]
        val = movielens.iloc[val_ids, :]
        assert set(train['UserID']) & set(val['UserID']) == set()
        train.to_csv(
            pathify('data', 'interim', 'movielens-cv{}-train.csv'.format(i)),
            index=False
        )
        val.to_csv(
            pathify('data', 'interim', 'movielens-cv{}-val.csv'.format(i)),
            index=False
        )


@timing
def preprocess(input_path, output_path, feature_names, label_name, num_categories):
    fields = [label_name] + feature_names
    with open(output_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for i, row in (iter_as_dict(input_path)):
            hashed_features = {
                label_name: row[label_name]
            }
            for feature in feature_names:
                str_to_hash = '{}-{}'.format(feature, row[feature])
                hashed_features[feature] = categorize_by_hash(str_to_hash, num_categories)
            writer.writerow(hashed_features)


@timing
def make(is_debug=False):
    movielens = pd.read_csv(pathify('data', 'interim', 'movielens.csv'))
    movielens = extract_features(movielens)
    movielens = make_binary_label(movielens)
    movielens = preprocess_features(movielens)
    movielens.to_csv(
        pathify('data', 'interim', 'movielens-train-test.csv'),
        index=False
    )
    split_for_validation(movielens)

    feature_names = 'UserID,MovieID,Age,Occupation,Action,Adventure,Animation,Children\'s,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western,debut_year'.split(',')
    label_name = 'Click'
    for i in range(5):
        train_input_path = pathify('data', 'interim', 'movielens-cv{}-train.csv'.format(i))
        val_input_path = pathify('data', 'interim', 'movielens-cv{}-val.csv'.format(i))
        train_output_path = pathify('data', 'processed', 'movielens-cv{}-train.csv'.format(i))
        val_output_path = pathify('data', 'processed', 'movielens-cv{}-val.csv'.format(i))

        preprocess(
            input_path=train_input_path,
            output_path=train_output_path,
            feature_names=feature_names,
            label_name=label_name,
            num_categories=2**5
        )
        preprocess(
            input_path=val_input_path,
            output_path=val_output_path,
            feature_names=feature_names,
            label_name=label_name,
            num_categories=2**5
        )
