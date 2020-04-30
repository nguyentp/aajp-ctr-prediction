# -*- coding: utf-8 -*-
from ajctr.features import build_features
def main():
    # build feature for avazu dataset
    # the processed files are generated in data/processed/avazu
    build_features.process_avazu(is_debug=False)


if __name__ == '__main__':
    main()
