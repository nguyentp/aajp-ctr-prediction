# -*- coding: utf-8 -*-
from ajctr.helpers import timing
from ajctr.features import make_movielens, make_avazu


@timing
def make(is_debug=False):
    make_movielens.make(is_debug)
    make_avazu.make(is_debug)
