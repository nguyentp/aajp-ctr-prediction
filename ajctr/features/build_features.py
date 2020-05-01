from ajctr.features.avazu import add_dummy_label, Preprocess_1, Preprocess_2
from ajctr.helpers import log, pathify, mkdir

def process_avazu(is_debug):
    """Full function to extract features from avazu's raw data
    
    """
    if is_debug:
        raw_train = pathify('data', 'raw', 'avazu', 'sample', 'train')
        raw_test = pathify('data', 'raw', 'avazu', 'sample', 'test')
    else:
        raw_train = pathify('data', 'raw', 'avazu', 'train')
        raw_test = pathify('data', 'raw', 'avazu', 'test')
    
    dummy_test = pathify('data', 'interim', 'avazu', 'dummy_test')
    add_dummy_label(raw_test, dummy_test)
    
    # add new features
    interim_train = pathify('data', 'interim', 'avazu', 'train')
    interim_test = pathify('data', 'interim', 'avazu', 'test')
    
    feature_gen = Preprocess_1()
    feature_gen.count_rows_per_feature(raw_train, raw_test)
    feature_gen.run(raw_train, interim_train, is_train=True)
    feature_gen.run(dummy_test, interim_test, is_train=False)
    
    # hashing features
    processed_train = pathify('data', 'processed', 'avazu', 'train')
    processed_test = pathify('data', 'processed', 'avazu', 'test')
    
    hashing = Preprocess_2(nr_thread=12, nr_bins=1000000)
    hashing.run(interim_train, processed_train)
    hashing.run(interim_test, processed_test)
    
    