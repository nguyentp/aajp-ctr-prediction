import pandas as pd
from sklearn.model_selection import GroupKFold


movielens = pd.read_csv('./data/interim/movielens-train-test.csv')
userids = movielens['UserID'].tolist()
movielens.shape
movielens.head()


kfold = GroupKFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kfold.split(movielens, groups=userids)):
    train = movielens.iloc[train_index, :]
    val = movielens.iloc[val_index, :]
    assert set(train['UserID']) & set(val['UserID']) == set()