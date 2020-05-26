"""Helpers module:
- Create log object
- Support some basic IO functions
"""
import os
import time
import shutil
import logging
import pickle
import functools
import csv
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style


__all__ = [
    'mkdir', 'timing',
    'pathify', 'listify', 
    'save_json', 'load_json', 
    '_pickle', 'load_pickle',
    'log', 'csv_reader', 'csv_writer'
]


class Log(object):
    # ALL INSTANCE AND SUBCLASS WILL SHARE THIS STATE.
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state

    def make(self, level=logging.INFO):
        if not hasattr(self, 'logger'):
            logger = logging.getLogger()
            logger.setLevel(level)
            if logger.hasHandlers(): logger.handlers = []
            fmt = logging.Formatter(f'[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
            hdl = logging.StreamHandler()
            hdl.setFormatter(fmt)
            logger.addHandler(hdl)
            self.logger = logger
        return self.logger


log = Log().make()


def mkdir(path, is_clean=False):
    try:
        if is_clean and os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    except IOError as e:
        log.info(e)


def pathify(*tokens):
    if len(tokens) == 0:
        raise ValueError('At least 1 token to create path')
    return os.path.abspath(os.path.join(*tokens))


def load_json(path, encoding=None):
    try:
        with open(path, encoding=encoding) as f:
            return json.load(f)
    except IOError as e:
        log.info(e)


def save_json(obj, path):
    try:
        with open(path, 'w') as f:
            json.dump(obj, f)
    except IOError as e:
        log.info(e)


def save_pickle(obj, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except IOError as e:
        log.info(e)


def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except IOError as e:
        log.info(e)


def load_processed_data(path, label_col):
    df = pd.read_csv(path, dtype=np.int16)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def timing(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        log.info(f'Start {fn.__module__} {fn.__name__}')
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        end = time.perf_counter()
        log.info(f'End {fn.__module__} {fn.__name__} in {end - start:.2f}s')
        return value
    return wrapper


def csv_reader(path, as_dict=True):
    if as_dict:
        return csv.DictReader(open(path))
    else:
        return csv.reader(open(path))

def csv_writer(path, headers=None, as_dict=True):
    if as_dict:
        return csv.DictWriter(open(path, 'w'), headers)
    else:
        return csv.writer(open(path, 'w'))


def categorize_by_hash(x, num_categories):
    """Using hashtrick to categorize input to fixed-size dimension.

    Returns:
    --------
        Return index from 0 to (num_categories - 1)
    """
    return int(hashlib.md5(str(x).encode('utf8')).hexdigest(), 16) % num_categories 


def iter_as_dict(path_to_file):
    with open(path_to_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            yield i, row


pd.options.display.max_columns = 999
style.use('fivethirtyeight')
