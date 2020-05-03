# -*- coding: utf-8 -*-
from ajctr.helpers import log
from ajctr.data import make_dataset
from ajctr.features import make_features
def main():
    log.info('1) Making dataset...')
    make_dataset.make()
    log.info('2) Making features...')
    make_features.make()

if __name__ == '__main__':
    main()
