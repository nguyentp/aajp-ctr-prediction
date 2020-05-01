# -*- coding: utf-8 -*-
from ajctr.data import make_dataset
from ajctr.features import make_features


def main():
    make_dataset.make()
    make_features.make()


if __name__ == '__main__':
    main()
