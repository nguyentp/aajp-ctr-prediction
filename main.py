# -*- coding: utf-8 -*-
from ajctr.helpers import timing, log
from ajctr.data import make_dataset
from ajctr.features import make_features
from ajctr.models import train_model


@timing
def main():
    make_dataset.make(is_debug=True)
    make_features.make(is_debug=True)
    train_model.train()


if __name__ == '__main__':
    main()
