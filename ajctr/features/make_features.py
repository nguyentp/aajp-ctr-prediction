# -*- coding: utf-8 -*-
from ajctr.helpers import timing
from ajctr.features import make_avazu


@timing
def make(is_debug=False):
    make_avazu.make(is_debug)
