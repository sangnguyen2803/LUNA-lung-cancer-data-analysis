# Kết quả tối ưu F1 cho mô hình kết hợp

Thí nghiệm trên 786 nốt của 435 bệnh nhân development. Dùng 5 outer fold cố định; mỗi outer training được tuning bằng 3 inner fold theo bệnh nhân.

Tiêu chí chọn cấu hình và ngưỡng là F1 cho MALIGN. Hai nhóm đầu vào đều có radiomics và tabular; một nhóm thêm log_volume, intensity_difference và mask_fill_ratio. Thử logistic regression, random forest và histogram gradient boosting.

| Phương pháp | F1 outer CV | SD F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Kết hợp cũ, ngưỡng 0.5 | 0.8312 | 0.0291 | 0.7958 | 0.8720 | 0.9344 |
| Kết hợp cũ, chọn ngưỡng trong inner CV | 0.8155 | 0.0239 | 0.7895 | 0.8473 | 0.9344 |
| Chọn model theo F1, ngưỡng 0.5 | 0.8106 | 0.0371 | 0.7937 | 0.8358 | 0.9238 |
| Chọn model và ngưỡng theo F1 | 0.8027 | 0.0301 | 0.7903 | 0.8220 | 0.9238 |

Quy trình chọn model/ngưỡng theo F1 có mean outer F1 0.8027, giảm 2.85 điểm phần trăm so với tham chiếu kết hợp cũ.

Cấu hình từng outer fold:

| Fold | Model | Nhóm feature | Ngưỡng |
|---|---|---|---:|
| 1 | hist_gradient_boosting | combined_curated | 0.250 |
| 2 | logistic | combined | 0.600 |
| 3 | logistic | combined | 0.575 |
| 4 | logistic | combined | 0.550 |
| 5 | logistic | combined_curated | 0.525 |

Candidate fit trên toàn bộ development: **logistic / combined_curated**, ngưỡng **0.500**, percentile 100, tham số `{"C": 0.1}`.

Candidate được lưu ở `artifacts/f1_optimization/f1_optimized_combined_model.joblib`. Ngưỡng phải được dùng khi chuyển score thành nhãn. Inner best score dùng để chọn cấu hình không phải hiệu năng đánh giá cuối cùng.

Outer CV đánh giá quy trình chọn cấu hình; model được chọn có thể khác giữa các fold. Tham chiếu cũ dùng tham số đã chọn trên development trước đây. Development cũng đã được dùng để định hướng các thử nghiệm trước, nên kết quả này vẫn là thăm dò, chưa chứng minh cải thiện có ý nghĩa thống kê hoặc hiệu năng trên dữ liệu độc lập.

Tập test cũ, model baseline và dự đoán test giữ nguyên, được kiểm tra bằng hash. Không so trực tiếp các số outer CV này với F1 test 85.27% của Random Forest tabular. Model mới chưa có test score độc lập.

Biểu đồ so sánh có trong output của [notebook 04](04_f1_optimization.ipynb). Khi chạy lại, ảnh được lưu vào `artifacts/f1_optimization/f1_optimization_comparison.png`.

Code và outputs: `04_f1_optimization.ipynb`. Kết quả từng fold, inner group assignments, search grids, outer OOF predictions và model/schema nằm trong `artifacts/f1_optimization/`.

## Đối chiếu cùng mức CV lồng nhau

Chọn lại cấu hình kết hợp trong từng outer training theo ROC-AUC, dùng đúng lưới sáu logistic và tám forest của thí nghiệm ban đầu, ngưỡng cố định 0.5. So với quy trình chọn model/ngưỡng theo F1, hai phương pháp có cùng outer validation và inner patient folds. Quy trình F1 có thêm boosting và nhóm feature curated, nên không phải hai không gian tìm kiếm giống nhau.

| Quy trình | F1 outer CV | SD F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Chọn model/ngưỡng theo F1 | 0.8027 | 0.0301 | 0.7903 | 0.8220 | 0.9238 |
| Chọn model theo ROC-AUC, ngưỡng 0.5 | 0.8278 | 0.0347 | 0.7925 | 0.8685 | 0.9299 |

Trên đối chiếu nested này, mean F1 của quy trình tối ưu F1 giảm 2.51 điểm phần trăm. Không dùng kết quả outer để sửa lại candidate grid hoặc chọn ngưỡng. Các số này vẫn là đánh giá trên development, chưa phải test độc lập.

Nhãn MALIGN/BENIGN dựa trên đánh giá của bác sĩ; chưa phải nhãn ung thư xác nhận bằng mô bệnh học.
