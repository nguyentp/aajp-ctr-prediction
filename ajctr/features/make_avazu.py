from collections import defaultdict
import hashlib
import math
import os 
import subprocess
from multiprocessing import Pool
import pandas as pd
import csv
from ajctr.helpers import log, csv_writer, csv_reader, pathify, timing


def make_output_headers():
    headers = 'id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21'.split(',')
    headers.remove('click')
    return ['click'] + headers


def make_userid_from_row(row):
    return '{}|{}'.format(row['device_ip'], row['device_model'])


def make_hour_from_row(row):
    # hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    return row['hour'][-2:]


def is_million(x):
    return (x + 1) % 10**6 == 0


def iter_as_dict(path_to_file):
    with open(path_to_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            yield i, row


def prepare_count_features(path_to_file):
    count_features = {}
    count_features['device_id_count'] = defaultdict(int)
    count_features['device_ip_count'] = defaultdict(int)
    count_features['user_id_count'] = defaultdict(int)
    count_features['hour_count'] = defaultdict(int)

    for i, row in iter_as_dict(path_to_file):
        count_features['device_id_count'][row['device_id']] += 1
        count_features['device_ip_count'][row['device_ip']] += 1
        count_features['user_id_count'][make_userid_from_row(row)] += 1
        count_features['hour_count'][make_hour_from_row(row)] += 1
        if is_million(i):
            log.info('Count {} mil.rows in {}'.format(i + 1, path_to_file))
    return count_features


def add_count_features_to_row(row, count_features):
    after_add = row.copy()
    after_add['device_id_count'] = (
        count_features['device_id_count'].get(row['device_id'], 0)
    )
    after_add['device_ip_count'] = (
        count_features['device_ip_count'].get(row['device_ip'], 0)
    )
    after_add['user_id_count'] = (
        count_features['user_id_count'].get(make_userid_from_row(row), 0)
    )
    after_add['hour_count'] = (
        count_features['hour_count'].get(make_hour_from_row(row), 0)
    )
    return after_add


def make_features(input_file, output_file, is_test=False):
    count_features = prepare_count_features(input_file)
    fields = make_output_headers() + list(count_features.keys())
    with open(output_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for i, row in (iter_as_dict(input_file)):
            if is_million(i):
                log.info('Write {} mil.rows to {}'.format(i + 1, output_file))
            row_to_write = add_count_features_to_row(row, count_features)
            if is_test:
                row_to_write['click'] = -1
            writer.writerow(row_to_write)


@timing
def make(is_debug=False):
    csv_folder = pathify('data', 'raw', 'avazu')
    if is_debug:
        csv_folder = pathify(csv_folder, 'sample')

    make_features(
        input_file=pathify(csv_folder, 'train'),
        output_file='data/interim/avazu-train.csv'
    )

    make_features(
        input_file=pathify(csv_folder, 'test'),
        output_file='data/interim/avazu-test.csv',
        is_test=True
    )
    

class Preprocess_1:
    """Module for adding new features to raw data
    
    New features added are:
    - counting features: reference function count_rows_per_feature
    - click history feautres: history each user click to any ads
    """
    def __init__(self):
        # all raw data's fields
        # EXCEPT 'pub_id', 'pub_domain', 'pub_category', added 'app/site_id', 'app/site_domain', 'app/site_category'
        self.fields = ['id','click','hour','banner_pos','device_id','device_ip','device_model','device_conn_type',
                       'C14','C17','C20','C21',
                       'app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']
        # raw data's fields and new features
        self.new_fields = self.fields +['device_id_count','device_ip_count','user_count','smooth_user_hour_count','user_click_histroy']
        
        # init count features for counting step in _count_rows_per_feature()
        self.id_cnt = collections.defaultdict(int)
        self.ip_cnt = collections.defaultdict(int)
        self.user_cnt = collections.defaultdict(int)
        self.user_hour_cnt = collections.defaultdict(int)
        
        # init click history features
        self.history = collections.defaultdict(lambda: {'history': '', 'buffer': '', 'prev_hour': ''})
        
    def run(self, src_path, dst_path, is_train):
        """Main method of this class

        Read existing features from src file,
        calculate new feature and update to dst file
        Agrs:
            src_path: raw input (train/test) files
            dst_path: output (train/test) files after adding features
        Returns:
            None
        """
        reader = csv_reader(src_path)
        writer = csv_writer(dst_path, headers=self.new_fields)
        writer.writeheader()

        # add new features for TRAIN file
        for i, row in enumerate(reader, start=1):
            new_row = {}
            for field in self.fields:
                new_row[field] = row[field]

            new_row['device_id_count'] = self.id_cnt[row['device_id']]
            new_row['device_ip_count'] = self.ip_cnt[row['device_ip']]

            user, hour = self._def_user(row), row['hour']
            new_row['user_count'] = self.user_cnt[user]
            new_row['smooth_user_hour_count'] = str(self.user_hour_cnt[user+'-'+hour])

            # add click history feature to this id only
            if row['device_id'] == 'a99f214a':

                if self.history[user]['prev_hour'] != row['hour']:
                    self.history[user]['history'] = (self.history[user]['history'] + self.history[user]['buffer'])[-4:]
                    self.history[user]['buffer'] = ''
                    self.history[user]['prev_hour'] = row['hour']

                new_row['user_click_histroy'] = self.history[user]['history']
                if is_train:
                    self.history[user]['buffer'] += row['click']
            else:
                new_row['user_click_histroy'] = ''

            writer.writerow(new_row)
            
    def count_rows_per_feature(self, train_path, test_path=None):
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
            None
        """
        # count rows for train data
        for i, row in enumerate(csv_reader(train_path), start=1):
            user = self._def_user(row)

            self.id_cnt[row['device_id']] += 1
            self.ip_cnt[row['device_ip']] += 1
            self.user_cnt[user] += 1
            self.user_hour_cnt[user+'-'+row['hour']] += 1
            
        if test_path != None:
            # count rows for test data
            for i, row in enumerate(csv_reader(test_path), start=1):
                user = self._def_user(row)

                self.id_cnt[row['device_id']] += 1
                self.ip_cnt[row['device_ip']] += 1
                self.user_cnt[user] += 1
                self.user_hour_cnt[user+'-'+row['hour']] += 1

    def _def_user(self, row):
        """Reference from 3 idiots's solution
        For specific device id, define user in different way
        """
        if row['device_id'] == 'a99f214a':
            user = 'ip-' + row['device_ip'] + '-' + row['device_model']
        else:
            user = 'id-' + row['device_id']

        return user

class Preprocess_2:
    """Module for converting category string features to an unique index number
    
    To make model be easier to access data feature,
    this module turns any string sequences into a number to represent for that category.
    For example: hash(site id-68fd1e64) => 839297, hash(site id-75fg1f15) => 420682
    Agrs:
        nr_thread: number of threads for parallelizing
        nr_bins: number of category after converting
    """
    def __init__(self, nr_thread, nr_bins):
        self.nr_thread = nr_thread
        self.nr_bins = nr_bins
        
        # features need to be convertedß
        self.fields = ['banner_pos','device_model','device_conn_type','C14','C17','C20','C21',
         'app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']

    def run(self, src_path, dst_path):
        """Main method of this class

        Conclude below steps:
        - split 1 large files into multiple small files
        - parallelly execute converting small files
        - merge converted small files to 1 large file
        - delete tmp small files during conversion
        Agrs:
            src_path: input file before converting
            dst_path: output file after converting
        Returns:
            None
        """
        self._split(src_path)
        
        self._parallel_convert(src_path, dst_path)
        
        self._cat(dst_path)

        self._delete(src_path)
        self._delete(dst_path)


    def _split(self, path, has_header=True):
        """Divide 1 large file into multiple small files
        Agrs:
            path: input large file
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
            return math.ceil(float(nr_lines)/self.nr_thread)

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

    def _parallel_convert(self, src_path, dst_path):
        """parallel execute convert script
        Agrs:
            src_path: path of file before convert
            dst_path: path of file after convert
        Returns:
            None
        """
        pool = Pool(processes=self.nr_thread)
        tasks = [(src_path + '.__tmp__.{0}'.format(i), 
                  dst_path + '.__tmp__.{0}'.format(i))
                 for i in range(self.nr_thread)
                ]
        pool.map(self._multi_run_wrapper, tasks)

    def _cat(self, path):
        """Merge multiple small files into 1 large file
        Agrs:
            path: input small files' location
            nr_thread: number of small files
        Returns:
            None
        """
        if os.path.exists(path):
            os.remove(path)
        for i in range(self.nr_thread):
            cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()

    def _delete(self, path):
        """delete tmp files
        """
        for i in range(self.nr_thread):
            os.remove('{0}.__tmp__.{1}'.format(path, i))

    def _multi_run_wrapper(self, args):
            return self._hash(*args)

    def _hash(self, src_path, dst_path):
        def hashstr(input):
            return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(self.nr_bins-1)+1)

        with open(dst_path, 'w') as f:
            for i, row in enumerate(csv_reader(src_path), start=1):
                feats = []

                for field in self.fields:
                    feats.append(hashstr(field+'-'+row[field]))
                feats.append(hashstr('hour-'+row['hour'][-2:]))

                feats.append(hashstr('device_id-'+row['device_id'] + '-' + row['device_id_count']))

                feats.append(hashstr('smooth_user_hour_count-'+row['smooth_user_hour_count']))

                feats.append(hashstr('user_click_histroy-'+row['user_count']+'-'+row['user_click_histroy']))

                f.write('{0} {1} {2}\n'.format(row['id'], row['click'], ' '.join(feats)))
                
                
def add_dummy_label(src_path, dst_path):
    """Add dummy label for test data
    
    Args:
        src_path: raw path for test data
        dst_path: output path with dummy label
    Returns:
        None
    """
    writer = csv_writer(dst_path, as_dict=False)
    for i, row in enumerate(csv_reader(src_path, as_dict=False)):
        if i == 0:
            row.insert(1, 'click')
        else:
            row.insert(1, '0')
        writer.writerow(row)
