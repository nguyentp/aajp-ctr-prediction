* baseline
  * 0.39663
* Modify user_count + history
  * 0.39671
```
#if int(row['user_count']) > 30:
#    feats.append(hashstr('user_click_histroy-'+row['user_count']))
#else:
#    feats.append(hashstr('user_click_histroy-'+row['user_count']+'-'+row['user_click_histroy']))
feats.append(hashstr('user_click_histroy-'+row['user_count']+'-'+row['user_click_histroy']))
```
* Modify smooth user count
  * 0.39673
```
#if int(row['smooth_user_hour_count']) > 30:
#    feats.append(hashstr('smooth_user_hour_count-0'))
#else:
#    feats.append(hashstr('smooth_user_hour_count-'+row['smooth_user_hour_count']))
feats.append(hashstr('smooth_user_hour_count-'+row['smooth_user_hour_count']))
```
* Modify device_id
  * 0.39648
```
#if int(row['device_id_count']) > 1000:
#    feats.append(hashstr('device_id-'+row['device_id']))
#else:
#    feats.append(hashstr('device_id-less-'+row['device_id_count']))
feats.append(hashstr('device_id-'+row['device_id'] + '-' + row['device_id_count']))
```
* Modify device_ip
  * 0.41758 -> this makes result get worse
```
#if int(row['device_ip_count']) > 1000:
#    feats.append(hashstr('device_ip-'+row['device_ip']))
#else:
#    feats.append(hashstr('device_ip-less-'+row['device_ip_count']))
feats.append(hashstr('device_ip-'+row['device_ip'] + '-' + row['device_ip_count']))
```
* Modify all except device_ip -> **choose this as final solution**
  * 0.39668
  * 0.38413 (with full data)