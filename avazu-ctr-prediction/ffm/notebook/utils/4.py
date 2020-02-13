#!/usr/bin/env python3

import argparse, csv, sys, pickle, collections

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path', type=str)
parser.add_argument('va_src_path', type=str)
parser.add_argument('tr_dst_path', type=str)
parser.add_argument('va_dst_path', type=str)
args = vars(parser.parse_args())

fields = ['banner_pos','device_model','device_conn_type','C14','C17','C20','C21',
         'app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']

def convert(src_path, dst_path, is_train):
    with open(dst_path, 'w') as f:
        for row in csv.DictReader(open(src_path)):
            
            feats = []

            for field in fields:
                feats.append(hashstr(field+'-'+row[field]))
            feats.append(hashstr('hour-'+row['hour'][-2:]))

            feats.append(hashstr('device_ip-' + row['device_ip'] + '-' + row['device_ip_count']))
            
            feats.append(hashstr('device_ip_count-' + row['device_ip_count']))

            feats.append(hashstr('device_id-' + row['device_id'] + '-' + row['device_id_count']))
            
            feats.append(hashstr('device_id_count-' + row['device_id_count']))

            feats.append(hashstr('smooth_user_hour_count-'+row['smooth_user_hour_count']))

            feats.append(hashstr('user_click_histroy-'+row['user_count']+'-'+row['user_click_histroy']))

            f.write('{0} {1} {2}\n'.format(row['id'], row['click'], ' '.join(feats)))

convert(args['tr_src_path'], args['tr_dst_path'], True)
convert(args['va_src_path'], args['va_dst_path'], False)
