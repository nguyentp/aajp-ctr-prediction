# -*- coding: utf-8 -*-
from ajctr.helpers import timing
from ajctr.features import make_movielens, make_avazu


@timing
def make():
    make_movielens.make()
    make_avazu.make()
