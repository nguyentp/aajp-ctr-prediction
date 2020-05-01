import os
from ajctr.features.avazu import add_dummy_label, Preprocess_1, Preprocess_2
def process_avazu(is_debug, raw_path='data/raw/avazu', processed_path='data/processed/avazu', interim_path='data/interim/avazu'):
    """Full function to extract features from avazu's raw data
    
    Add new features and implement hashing for raw data.
    The processed datas are saved into 'data/processed'
    Args:
        raw_path: path to avazu's raw data include train and test set
        processed_path: path where processed datas are saved
        interim_path: path where interim files are saved during processing
    Returns:
        None
    """
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(interim_path, exist_ok=True)
    if is_debug:
        raw_train = os.path.join(raw_path, 'sample/train')
        raw_test = os.path.join(raw_path, 'sample/test')
    else:
        raw_train = os.path.join(raw_path, 'train')
        raw_test = os.path.join(raw_path, 'test')
    
    add_dummy_label(raw_test, os.path.join(interim_path, 'dummy_test'))
    raw_test = os.path.join(interim_path, 'dummy_test')
    
    # add new features
    interim_train = os.path.join(interim_path, 'train')
    interim_test = os.path.join(interim_path, 'test')
    feature_gen = Preprocess_1()
    feature_gen.count_rows_per_feature(raw_train, raw_test)
    feature_gen.run(raw_train, interim_train, is_train=True)
    feature_gen.run(raw_test, interim_test, is_train=False)
    
    # hashing features
    processed_train = os.path.join(processed_path, 'train')
    processed_test = os.path.join(processed_path, 'test')
    hashing = Preprocess_2(nr_thread=12, nr_bins=1000000)
    hashing.run(interim_train, processed_train)
    hashing.run(interim_test, processed_test)
    
    