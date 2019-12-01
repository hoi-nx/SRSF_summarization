# Giới thiệu

Mô hình này kết hợp sentence feature và deeplearning

## Quickstart

### Step 1: Cài đặt môi trường

1. Mã nguồn này được thử nghiệm trên Ubuntu 16.04 LTS
2. Yêu cầu: python 3.6 trở lên
3. Cài đặt các thư viện python liên quan: pip install -r requirements.txt


### Step 2: Huấn luyện cho mô hình
#### Model
1. RNN_RNN : Mô hình gốc Summarunner run train and test with run.py
2. SRS2F_RNN_RNN : Mô hình kết hợp thêm 2 feature: Độ tương đồng với câu chủ đề và pagerank. Run train and test with run.py
3. SRSF_RNN_RNN_V2: Mô hình kết hợp thêm 6 feature. Run train and test with main_v2.py
4.  SRSF_RNN_RNN_V4: Mô hình kết hợp thêm 9 feature. Run train and test with main_v3.py

### Step 3: Chạy test
1. python3 main_v2.py -batch_size 32 -hyp [path_save_out_put] -test -test_dir [path_input_test] -load_dir [path_model]
2. python3 main_v3.py -batch_size 32 -hyp [path_save_out_put] -test -test_dir [path_input_test] -load_dir [path_model]
3. python3 run.py -batch_size 32 -hyp [path_save_out_put] -test -test_dir [path_input_test] -load_dir [path_model]

### Step 4: Predict

### Step 5: Đánh giá bằng Rouge


## References:
Summarunner: 
