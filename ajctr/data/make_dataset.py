# -*- coding: utf-8 -*-
from ajctr.helpers import timing
from ajctr.data import make_movielens


@timing
def make(is_debug=False):
    make_movielens.make(is_debug)
