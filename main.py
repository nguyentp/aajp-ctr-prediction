# -*- coding: utf-8 -*-
from ajctr.features import build_features
from ajctr.data import make_dataset
from ajctr.features import make_features
def main():
    # build feature for avazu dataset
    # the processed files are generated in data/processed/avazu
    build_features.process_avazu(is_debug=False)
    make_dataset.make()
    make_features.make()

if __name__ == '__main__':
    main()
