# -*- coding: utf-8 -*-
from ajctr.helpers import timing, log, mkdir, pathify
from ajctr.data import make_dataset
from ajctr.features import make_features
from ajctr.models import train_model


mkdir(pathify('data', 'interim'))
mkdir(pathify('data', 'processed'))

@timing
def main():
    make_dataset.make(is_debug=False)
    make_features.make(is_debug=False)
    train_model.train()


if __name__ == '__main__':
    main()
