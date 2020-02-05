# Data flow understanding Study
Tóm gọn lại, họ sẽ dùng tất cả các feature có trong data **(trừ C1~C13, C15, C16, C18, C19)** và các feature mới tự generate thêm. Data preprocessing của họ trước khi đưa vào training gồm các bước như sau:
## 1) Bước 1: Tạo thêm các feature mới (xử lý trên 1 thread)
#### *chỗ reference: util/gen_data.py*
ngoài [feature có sẵn](https://www.kaggle.com/c/avazu-ctr-prediction/data), họ thêm những feature mới sau:
* `pub_id`,`pub_domain`,`pub_category`
  * là feature gộp lại của `site_id`-`app_id`, `site_domain`-`app_domain`, `site_category`-`app_category`
  * nếu `row['site_id'] == '85f751fd'` thì `pub` đc lấy theo thông tin `app`, còn ko thì lấy theo thông tin `site`
  * `pub` đc lấy theo `site`-`app` thì lưu vào 2 file `site`-`app` riêng (Eg: `tr.0.site.csv`, `tr.0.app.csv`) để train 2 model
* Counting features
  * `device_id_count`: tổng số lượng `device_id` đó xuất hiện trên toàn tập data
  * `device_ip_count`: tương tự `device_id_count`
  * `user_count`: tương tự như trên. Tuy nhiên khác vs 2 feature trên là có sẵn. feature `user` ko có sẵn và đc định nghĩa như sau:
  ```
  if row['device_id'] == 'a99f214a':
      user = 'ip-' + row['device_ip'] + '-' + row['device_model']
  else:
      user = 'id-' + row['device_id']
  ```
  -> *cách này quá specific trong trường hợp cụ thể nên ko áp dụng cho định hướng generalize của mình đc, hoặc mình cần đưa ra 1 định nghĩa chung cho `user`*
  * `smooth_user_count_hour`: tương tự `user_count` chỉ khác ở việc là tính count theo `hour`
  -> *ko áp dụng đc*
  
* Click History
  * Chỉ áp dụng cho row có `device_id` == "a99f214a" ???. Quote from document của họ: "We generate a click history feature for users who have device id information"
  * History theo từng `user`
  * Lưu theo kiểu string `0110` theo thứ tự thời gian từ trái qua phải
    * `0`: adv đó xuất hiện mà user ko click
    * `1`: adv đó xuất hiện mà user click
  * Chỉ lấy 4 records gần nhất của click history `hour` ngay phía trước
## 2) Bước 2: hashing **1 số** features (xử lý multithread)
#### *chỗ reference: converter/2.py*
* hashing với md5 rồi lấy 6 chữ số cuối hash(site id-68fd1e64) => 739920192382357**839297**
```
NR_BINS = 1000000
str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)
```
* cú pháp của string cho hashing: feature_name-feature_value: Eg: 'site_id-68fd1e64'
* hầu hết hash các features đều làm theo cú pháp trên. Tuy nhiên 1 số lại đc xử lý riêng như sau:
  * feature `hour`: string khi hash là 'hour-2 ký tự cuối'. Eg: row['hour'] = '14103100' -> 'hour-00'
  * sử dụng count feature để tách thành 2 nhóm hashing
  ```
  if int(row['device_ip_count']) > 1000:
      feats.append(hashstr('device_ip-'+row['device_ip']))
  else:
      feats.append(hashstr('device_ip-less-'+row['device_ip_count']))
  ```

## Sau khi qua bước hashing ta sẽ được 17 con số, data sẽ trông như thế này trước khi cho vào training
```
10002518649031436658 0 252610 14756 301965 534902 566915 416678 875413 903024 536024 712135 287432 860490 156287 600895 164124
```
theo thứ tự như sau:
1. id
2. click 
3. pub_id 
4. pub_domain 
5. pub_category 
6. banner_pos 
7. device_model 
8. device_conn_type 
9. C14 
10. C17 
11. C20 
12. C21 
13. hour 
14. device_ip 
15. device_id 
16. smooth_user 
17. click_history
