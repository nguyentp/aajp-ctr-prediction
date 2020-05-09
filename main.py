# -*- coding: utf-8 -*-
import argparse
from ajctr.helpers import timing, log
from ajctr.data import make_dataset
from ajctr.features import make_features
from ajctr.models import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--is-debug', default=False, type=bool)
args = parser.parse_args('')
if args.is_debug:
    log.info('Run program in Debug mode')


@timing
def main():
    make_dataset.make(args.is_debug)
    make_features.make(args.is_debug)

    train_model.train()

if __name__ == '__main__':
    main()
