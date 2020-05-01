import csv
import collections
import hashlib
import math
import os 
import subprocess
from multiprocessing import Pool
import hashlib

def add_dummy_label(src_path, dst_path):
    """Add dummy label for test data
    
    Args:
        src_path: raw path for test data
        dst_path: output path with dummy label
    Returns:
        None
    """
    f = csv.writer(open(dst_path, 'w'))
    for i, row in enumerate(csv.reader(open(src_path))):
        if i == 0:
            row.insert(1, 'click')
        else:
            row.insert(1, '0')
        f.writerow(row)
        
def gen_new_features(src_train_path, src_test_path, dst_train_path, dst_test_path):
    """Add new features to raw data
    
    New features added are:
    - counting features: reference function count_rows_per_feature
    - click history feautres: history each user click to any ads
    Agrs:
        src_train/test_path: raw input (train/test) files
        dst_train/test_path: output (train/test) files after adding features
    Returns:
        None
    """
    # all raw data's fields
    # EXCEPT 'pub_id', 'pub_domain', 'pub_category', added 'app/site_id', 'app/site_domain', 'app/site_category'
    FIELDS = ['id','click','hour','banner_pos','device_id','device_ip','device_model','device_conn_type','C14','C17','C20','C21',
             'app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']
    # raw data's fields and new features
    NEW_FIELDS = FIELDS+['device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']
    
    history = collections.defaultdict(lambda: {'history': '', 'buffer': '', 'prev_hour': ''})    
    id_cnt, ip_cnt, user_cnt, user_hour_cnt = _count_rows_per_feature(src_train_path, src_test_path)
    
    reader = csv.DictReader(open(src_train_path))
    writer = csv.DictWriter(open(dst_train_path, 'w'), NEW_FIELDS)
    writer.writeheader()

    # add new features for TRAIN file
    for i, row in enumerate(reader, start=1):
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        new_row['device_id_count'] = id_cnt[row['device_id']]
        new_row['device_ip_count'] = ip_cnt[row['device_ip']]

        user, hour = _def_user(row), row['hour']
        new_row['user_count'] = user_cnt[user]
        new_row['smooth_user_hour_count'] = str(user_hour_cnt[user+'-'+hour])

        # add click history feature to this id only
        if row['device_id'] == 'a99f214a':

            if history[user]['prev_hour'] != row['hour']:
                history[user]['history'] = (history[user]['history'] + history[user]['buffer'])[-4:]
                history[user]['buffer'] = ''
                history[user]['prev_hour'] = row['hour']

            new_row['user_click_histroy'] = history[user]['history']

            history[user]['buffer'] += row['click']
        else:
            new_row['user_click_histroy'] = ''
            
        writer.writerow(new_row)
    
    # add new features for TEST file
    reader = csv.DictReader(open(src_test_path))
    writer = csv.DictWriter(open(dst_test_path, 'w'), NEW_FIELDS)
    writer.writeheader()

    for i, row in enumerate(reader, start=1):
        new_row = {}
        for field in FIELDS:
            new_row[field] = row[field]

        new_row['device_id_count'] = id_cnt[row['device_id']]
        new_row['device_ip_count'] = ip_cnt[row['device_ip']]

        user, hour = _def_user(row), row['hour']
        new_row['user_count'] = user_cnt[user]
        new_row['smooth_user_hour_count'] = str(user_hour_cnt[user+'-'+hour])

        # add click history feature to this id only
        if row['device_id'] == 'a99f214a':

            if history[user]['prev_hour'] != row['hour']:
                history[user]['history'] = (history[user]['history'] + history[user]['buffer'])[-4:]
                history[user]['buffer'] = ''
                history[user]['prev_hour'] = row['hour']

            new_row['user_click_histroy'] = history[user]['history']

        else:
            new_row['user_click_histroy'] = ''
            
        writer.writerow(new_row)


def hash_features(src_train_path, src_test_path, dst_train_path, dst_test_path):
    """Convert category string features to an unique index number
    
    To make model be easier to access data feature,
    this function turns any string sequences into a number to represent for that category.
    For example: hash(site id-68fd1e64) => 839297, hash(site id-75fg1f15) => 420682
    Agrs:
        src_train/test_path: input (train/test) file before hashing
        dst_train/test_path: output (train/test) file after hashing
    Returns:
        None
    """
    nr_thread = 12
    
    # split 1 src file into nr_thread files
    _split(path=src_train_path, nr_thread=nr_thread)
    _split(path=src_test_path, nr_thread=nr_thread)
    
    # parallelly hashing splited files and save to nr_thread hashed files
    _parallel_convert(src_train_path, dst_train_path, nr_thread)
    _parallel_convert(src_test_path, dst_test_path, nr_thread)
    
    # delete old splited src files
    _delete(src_train_path, nr_thread)
    _delete(src_test_path, nr_thread)

    # merge nr_thread dst files into 1 file
    _cat(dst_train_path, nr_thread)
    _cat(dst_test_path, nr_thread)

    # delete old splited dst csv_files
    _delete(dst_train_path, nr_thread)
    _delete(dst_test_path, nr_thread)

def _count_rows_per_feature(train_path, test_path):
    """Count number of rows with same ip address
    
    For whole train and test data set,
    count number of rows that have the same for 
    - id features
    - ip features
    - users (Define user as combination of device ip and device model)
    - users per hours
    Agrs:
        train_path: raw train data path
        test_path: raw test data path
    Returns:
        a tuple of 4 dictionaries corresponding with above 4 counts
    """
    id_cnt = collections.defaultdict(int)
    ip_cnt = collections.defaultdict(int)
    user_cnt = collections.defaultdict(int)
    user_hour_cnt = collections.defaultdict(int)
    
    # count rows for train data
    for i, row in enumerate(csv.DictReader(open(train_path)), start=1):
        user = _def_user(row)
            
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1
        user_cnt[user] += 1
        user_hour_cnt[user+'-'+row['hour']] += 1
        
    # count rows for test data
    for i, row in enumerate(csv.DictReader(open(test_path)), start=1):
        user = _def_user(row)
        
        id_cnt[row['device_id']] += 1
        ip_cnt[row['device_ip']] += 1
        user_cnt[user] += 1
        user_hour_cnt[user+'-'+row['hour']] += 1
    
    return (id_cnt, ip_cnt, user_cnt, user_hour_cnt)

def _def_user(row):
    """Reference from 3 idiots's solution
    For specific device id, define user in different way
    """
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']

    return user

def _split(path, nr_thread, has_header=True):
    """Divide 1 large file into multiple small files
    Agrs:
        path: input large file
        nr_thread: number of small files
        has_header: input file has header or not
    Returns:
        None
    """
    def open_with_first_line_skipped(path, skip=True):
        f = open(path)
        if not skip:
            return f
        next(f)
        return f

    def open_with_header_written(path, idx, header):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if not has_header:
            return f 
        f.write(header)
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        if not has_header:
            nr_lines += 1 
        return math.ceil(float(nr_lines)/nr_thread)

    header = open(path).readline()

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_written(path, idx, header)
    for i, line in enumerate(open_with_first_line_skipped(path, has_header), start=1):
        if i%nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_written(path, idx, header)
        f.write(line)
    f.close()
    
def _parallel_convert(src_path, dst_path, nr_thread):
    """parallel execute convert script
    Agrs:
        src_path: path of file before convert
        dst_path: path of file after convert
        nr_thread: number of paralleling convert
    Returns:
        None
    """
    pool = Pool(processes=nr_thread)
    tasks = [(src_path + '.__tmp__.{0}'.format(i), 
              dst_path + '.__tmp__.{0}'.format(i))
             for i in range(nr_thread)
            ]
    pool.map(multi_run_wrapper, tasks)
        
def _cat(path, nr_thread):
    """Merge multiple small files into 1 large file
    Agrs:
        path: input small files' location
        nr_thread: number of small files
    Returns:
        None
    """
    if os.path.exists(path):
        os.remove(path)
    for i in range(nr_thread):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

def _delete(path, nr_thread):
    """delete tmp files
    """
    for i in range(nr_thread):
        os.remove('{0}.__tmp__.{1}'.format(path, i))

def multi_run_wrapper(args):
        return convert(*args)
    
def hashstr(input, nr_bins=10000):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1)

def convert(src_path, dst_path):
    fields = ['banner_pos','device_model','device_conn_type','C14','C17','C20','C21',
         'app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']

    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            
            feats = []

            for field in fields:
                feats.append(hashstr(field+'-'+row[field]))
            feats.append(hashstr('hour-'+row['hour'][-2:]))

            feats.append(hashstr('device_id-'+row['device_id'] + '-' + row['device_id_count']))
            
            feats.append(hashstr('smooth_user_hour_count-'+row['smooth_user_hour_count']))
            
            feats.append(hashstr('user_click_histroy-'+row['user_count']+'-'+row['user_click_histroy']))
            
            f.write('{0} {1} {2}\n'.format(row['id'], row['click'], ' '.join(feats)))