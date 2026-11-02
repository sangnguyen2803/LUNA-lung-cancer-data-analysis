# Kết quả thử thêm feature engineering

Thử tám đặc trưng tự tạo trên 786 nốt của 435 bệnh nhân development, dùng đúng 5 fold đã lưu.

Các đặc trưng: `log_volume`, `intensity_difference`, `inside_median`, `inside_iqr`, `mask_size_x_mm`, `mask_size_y_mm`, `mask_size_z_mm`, `mask_fill_ratio`.

Bảng sau so sánh cấu hình tốt nhất trước/sau, cùng ngân sách tìm kiếm và tiêu chí chọn ROC-AUC. F1 là trung bình CV cho MALIGN ở decision rule mặc định, không phải F1 test.

| Nhóm | Model trước → sau | AUC trước | AUC sau | Δ AUC | F1 trước | F1 sau | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tabular | random_forest → logistic | 0.9420 | 0.9399 | -0.0022 | 0.8281 | 0.8250 | -0.0031 |
| radiomics | random_forest → random_forest | 0.9271 | 0.9290 | +0.0019 | 0.8037 | 0.8059 | +0.0021 |
| combined | logistic → logistic | 0.9344 | 0.9357 | +0.0013 | 0.8312 | 0.8264 | -0.0048 |

So sánh giữ nguyên cấu hình baseline trước khi thêm đặc trưng:

| Nhóm | Model cố định | Δ AUC CV | Δ F1 CV |
|---|---|---:|---:|
| tabular | random_forest | -0.0093 | -0.0110 |
| radiomics | random_forest | -0.0042 | +0.0033 |
| combined | logistic | +0.0013 | -0.0048 |

Diễn giải theo từng nhóm:

- tabular: AUC giảm 0.0022; F1 giảm 0.0031. AUC cao hơn ở 2/5 fold; F1 cao hơn ở 2/5 fold.
- radiomics: AUC tăng 0.0019; F1 tăng 0.0021. AUC cao hơn ở 4/5 fold; F1 cao hơn ở 3/5 fold.
- combined: AUC tăng 0.0013; F1 giảm 0.0048. AUC cao hơn ở 5/5 fold; F1 cao hơn ở 1/5 fold.

Các điểm này là ước lượng CV được dùng để chọn cấu hình; chưa chứng minh khác biệt có ý nghĩa thống kê hoặc khả năng tổng quát hóa trên dữ liệu độc lập. Một số đặc trưng mới chồng lặp với radiomics, nên thêm đặc trưng có thể cải thiện một chỉ số và làm giảm chỉ số khác.

Tập test không được đánh giá lại. Các model và kết quả test ban đầu giữ nguyên, được xác nhận bằng hash. Vì đã xem test trước khi thiết kế thí nghiệm này, một kết luận hiệu năng cuối cùng cho model cải tiến cần đánh giá độc lập.

Biểu đồ so sánh có trong output của [notebook 03](03_feature_engineering_ablation.ipynb). Khi chạy lại, ảnh được lưu vào `artifacts/feature_engineering/feature_engineering_comparison.png`.

Kết quả chi tiết, model và fold differences nằm trong `artifacts/feature_engineering/`. Code và outputs nằm trong `03_feature_engineering_ablation.ipynb`.
