* Bước 1: Tạo thêm các feature mới (1 thread only)
*file code chính: util/gen_data.py*
  * ngoài những feature cũ, họ thêm những feature mới sau:
    
    (1) Counting features
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
    
    (2) Click History
    * Chỉ áp dụng cho row có `device_id` == "a99f214a" ???. Quote from document của họ: "We generate a click history feature for users who have device id information"
    * History theo từng `user`
    * Lưu theo kiểu string `0110` theo thứ tự thời gian từ trái qua phải
      * `0`: adv đó xuất hiện mà user ko click
      * `1`: adv đó xuất hiện mà user click
    * Chỉ lấy click history của `hour` ngay phía trước và 4 records
* Bước 2: hashing **1 số** features với xử lý multithread
*file code chính: converter/2.py*
  * hashing bằng md5: 
  ```
  NR_BINS = 1000000
  str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)
  ```
  * những 