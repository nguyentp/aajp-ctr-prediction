from collections import defaultdict
import hashlib
import math
import os 
import subprocess
import csv
from multiprocessing import Pool
import pandas as pd
from ajctr.helpers import (
    log, csv_writer, csv_reader, pathify,
    timing, categorize_by_hash, iter_as_dict,
    load_pickle, save_pickle
)


def make_output_headers():
    headers = 'id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21'.split(',')
    headers.remove('click')
    return ['click'] + headers


def make_userid_from_row(row):
    return '{}|{}'.format(row['device_ip'], row['device_model'])


def make_hour_from_row(row):
    # hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    return row['hour'][-2:]


def is_million(x):
    return (x + 1) % 10**6 == 0


def prepare_count_features(path_to_file):
    count_features = {}
    count_features['device_id_count'] = defaultdict(int)
    count_features['device_ip_count'] = defaultdict(int)
    count_features['user_id_count'] = defaultdict(int)
    count_features['hour_count'] = defaultdict(int)

    for i, row in iter_as_dict(path_to_file):
        count_features['device_id_count'][row['device_id']] += 1
        count_features['device_ip_count'][row['device_ip']] += 1
        count_features['user_id_count'][make_userid_from_row(row)] += 1
        count_features['hour_count'][make_hour_from_row(row)] += 1
        if is_million(i):
            log.info('Count {} mil.rows in {}'.format(i + 1, path_to_file))
    return count_features


def add_count_features_to_row(row, count_features):
    after_add = row.copy()
    after_add['device_id_count'] = (
        count_features['device_id_count'].get(row['device_id'], 0)
    )
    after_add['device_ip_count'] = (
        count_features['device_ip_count'].get(row['device_ip'], 0)
    )
    after_add['user_id_count'] = (
        count_features['user_id_count'].get(make_userid_from_row(row), 0)
    )
    after_add['hour_count'] = (
        count_features['hour_count'].get(make_hour_from_row(row), 0)
    )
    return after_add


def make_features(input_file, output_file, mode):
    count_filename = pathify('data', 'interim', 'avazu-cv-train-count-features.pickle')
    if mode in ['test', 'val']:
        count_features = load_pickle(count_filename)
    else:
        count_features = prepare_count_features(input_file)
        save_pickle(count_features, count_filename)

    fields = make_output_headers() + list(count_features.keys())
    with open(output_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for i, row in (iter_as_dict(input_file)):
            if is_million(i):
                log.info('Write {} mil.rows to {}'.format(i + 1, output_file))
            row_to_write = add_count_features_to_row(row, count_features)
            row_to_write['hour'] = make_hour_from_row(row)
            if mode == 'test':
                row_to_write['click'] = -1
            writer.writerow(row_to_write)


def split_for_validation(train_filename, is_debug):
    # Use date 30 in train data as validation data
    date_val = '141030'
    fields = 'id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21,device_id_count,device_ip_count,user_id_count,hour_count\n'
    cv_train_path = 'data/interim/avazu-train.csv'
    cv_val_path = 'data/interim/avazu-val.csv'

    with open(cv_train_path, 'w') as train_file:
        train_file.write(fields)
    with open(cv_val_path, 'w') as val_file:
        val_file.write(fields)

    with open(train_filename) as csv_file:
        with open(cv_train_path, 'a') as train_file:
            with open(cv_val_path, 'a') as val_file:
                for i, line in enumerate(csv_file):
                    if i == 0:
                        continue
                    if is_debug:
                        val_file.write(line)
                        train_file.write(line)
                    else:
                        if line.split(',')[2][:-2] == date_val:
                            val_file.write(line)
                        else:
                            train_file.write(line)

                    if is_million(i):
                        log.info('Splited {} mil.rows'.format(i + 1))


def preprocess(input_path, output_path, feature_names, label_name, num_categories):
    fields = [label_name] + feature_names
    with open(output_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for i, row in (iter_as_dict(input_path)):
            if is_million(i):
                log.info('Preprocessed {} mil.rows'.format(i + 1))
            hashed_features = {
                label_name: row[label_name]
            }
            for feature in feature_names:
                str_to_hash = '{}-{}'.format(feature, row[feature])
                hashed_features[feature] = categorize_by_hash(str_to_hash, num_categories)
            writer.writerow(hashed_features)


@timing
def make(is_debug=False):
    csv_folder = pathify('data', 'raw', 'avazu')
    if is_debug:
        csv_folder = pathify(csv_folder, 'sample')

    split_for_validation(pathify(csv_folder, 'train'), is_debug)

    make_features(
        input_file=pathify('data', 'interim', 'avazu-train.csv'),
        output_file=pathify('data', 'interim', 'avazu-train-feature.csv'),
        mode='train'
    )

    make_features(
        input_file=pathify('data', 'interim', 'avazu-val.csv'),
        output_file=pathify('data', 'interim', 'avazu-val-feature.csv'),
        mode='val'
    )

    feature_names = 'C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21,device_id_count,device_ip_count,user_id_count,hour_count'.split(',')
    label_name = 'click'
    preprocess(
        input_path=pathify('data', 'interim', 'avazu-train-feature.csv'),
        output_path=pathify('data', 'processed', 'avazu-cv-train.csv'),
        feature_names=feature_names,
        label_name=label_name,
        num_categories=2**16
    )
    preprocess(
        input_path=pathify('data', 'interim', 'avazu-val-feature.csv'),
        output_path=pathify('data', 'processed', 'avazu-cv-val.csv'),
        feature_names=feature_names,
        label_name=label_name,
        num_categories=2**16
    )
