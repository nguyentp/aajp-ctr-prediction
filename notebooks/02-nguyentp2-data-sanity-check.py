import os
import pandas as pd


print('curdir', os.path.abspath(os.curdir))
print(pd.__version__)

df = pd.read_csv('../data/raw/avazu/train', usecols=['hour', 'click'])

print(df.to_markdown())

df.head()

df['date'] = df['hour']//100
df.groupby('date').size().mean()

df['click'].sum() / len(df)

df = pd.read_csv('../data/raw/avazu/train', nrows=3)
print(df.to_markdown())

stats = []
for column in ['C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']:
    df = pd.read_csv('../data/raw/avazu/train', usecols=[column])
    stats.append({
        'Column': column,
        '#Unique': df[column].nunique(), 
        '#Unique / lengh(data)': df[column].nunique()/len(df)
    })
print(pd.DataFrame(stats)[['Column', '#Unique']].to_markdown())

    