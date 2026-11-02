# Báo cáo kết quả phân loại BENIGN / MALIGN

Báo cáo này tổng hợp kết quả đã chạy trong `01_voi_radiomics.ipynb` và `02_malign_benign_modeling.ipynb`. Mô hình được chọn bằng cross-validation là **Random Forest sử dụng tabular features**. Trên tập test giữ riêng theo bệnh nhân, mô hình đạt **ROC-AUC 0.9597** và **accuracy 90.87%**. Mô hình kết hợp radiomics và tabular cũng đã được huấn luyện và lưu; cấu hình kết hợp tốt nhất là Logistic Regression, với ROC-AUC trung bình trên CV **0.9344**.

## 1. Mục tiêu và dữ liệu

Mục tiêu là dự đoán nhãn BENIGN = 0 và MALIGN = 1 cho từng nốt phổi. Đầu vào gồm ảnh VOI 3D, mask đánh dấu nốt phổi và metadata của nốt. Đơn vị quan sát là một nốt phổi, không phải một lát cắt ảnh.

| Dữ liệu | Số lượng |
|---|---:|
| Nốt trong bảng metadata | 996 |
| Nốt có đầy đủ VOI và mask, trích đặc trưng thành công | 994 |
| BENIGN trong tập dùng để mô hình hóa | 650 |
| MALIGN trong tập dùng để mô hình hóa | 344 |
| Bệnh nhân trong tập dùng để mô hình hóa | 551 |
| Nốt bị loại vì thiếu ảnh VOI | 2 |

Hai ảnh thiếu là `LIDC-IDRI-1002_R_1.nii.gz` và `LIDC-IDRI-1002_R_2.nii.gz`; cả hai có nhãn MALIGN. Các trường hợp này được ghi nhận trong báo cáo loại trừ, không được thay thế bằng ảnh giả hoặc đưa vào riêng tập tabular khi so sánh các nhóm đặc trưng.

Nhãn `Diagnosis` của bộ dữ liệu được tạo từ đánh giá của bác sĩ: một nốt được gán MALIGN khi có ít nhất hai bản ghi annotation mang `Malignancy_value > 3`. Quy tắc này đã được kiểm tra với cả 996 dòng metadata bằng khóa `patient_id`, `nodule_id` và `seriesuid`. Đây là **nhãn đánh giá của bác sĩ, không phải kết quả ung thư đã được xác nhận bằng mô bệnh học**.

## 2. Tiền xử lý và trích đặc trưng

Ảnh và mask được kiểm tra kích thước, spacing, origin, direction, giá trị hữu hạn và tính nhị phân của mask. Các cặp ảnh được đưa về hướng LPS và resample về spacing 1 × 1 × 1 mm. Nội suy tuyến tính được dùng thống nhất cho ảnh; nearest-neighbor được dùng cho mask. Lựa chọn tuyến tính dựa trên kiểm tra chất lượng: một số VOI hai lát cắt gặp lỗi khi trích với B-spline trong môi trường hiện tại, nhưng cùng dữ liệu chạy thành công với nội suy tuyến tính.

Toàn bộ vùng có nhãn 1 được giữ lại, gồm cả các thành phần mask rời nhau. Các mask cần xem xét được đánh dấu trong báo cáo QC. Một cặp ảnh vượt qua kiểm tra hình học vẫn cần kiểm tra chất lượng đường viền segmentation.

PyRadiomics trích **107 đặc trưng** từ ảnh Original:

| Nhóm đặc trưng | Số đặc trưng |
|---|---:|
| Shape | 14 |
| First-order | 18 |
| GLCM | 24 |
| GLRLM | 16 |
| GLSZM | 16 |
| GLDM | 14 |
| NGTDM | 5 |
| Tổng | 107 |

Bin width được cố định bằng 25 đơn vị cường độ ảnh đã lưu. Việc hiệu chuẩn sang HU chưa được xác nhận, nên pipeline không mặc định rằng các giá trị này là HU. Không áp dụng normalization cường độ hoặc clipping theo HU. Cấu hình đầy đủ nằm trong `radiomics_params.yaml`.

Kết quả cuối cùng: **994/994 cặp ảnh có sẵn trích thành công**, không có giá trị thiếu trong bảng radiomics và không còn lỗi extraction. Bộ chín kiểm tra về hình học, mask, đầu vào inference, worker và bốn VOI mỏng đều đạt.

## 3. Các nhóm đầu vào được so sánh

| Nhóm | Thành phần | Số cột đầu vào thô |
|---|---|---:|
| Tabular | Đường kính và tám mô tả nốt phổi không thuộc đánh giá malignancy | 9 |
| Radiomics | Các đặc trưng trích từ VOI/mask | 107 |
| Kết hợp | Tabular và radiomics | 116 |

Tám mô tả tabular gồm Calcification, InternalStructure, Lobulation, Margin, Sphericity, Spiculation, Subtlety và Texture. Calcification và InternalStructure được one-hot encode; các điểm đánh giá có thứ tự được xử lý như biến số. Số cột sau encoding có thể khác số cột đầu vào thô. Hai bảng hiện có không chứa tuổi, giới tính hoặc tiền sử lâm sàng.

Các cột `Diagnosis`, `Diagnosis_value`, `Malignancy`, `Malignancy_value`, số phiếu malignancy và `len_mal_details` không được dùng làm predictors. Patient ID, series ID, nodule ID, tên file và tọa độ tuyệt đối cũng được loại khỏi đầu vào mô hình.

Ba nhóm đặc trưng dùng cùng 994 nốt và cùng cách chia bệnh nhân. Nhờ vậy, so sánh CV không bị ảnh hưởng bởi việc một nhóm có thêm các nốt thiếu ảnh.

## 4. Chia dữ liệu và lựa chọn mô hình

| Tập | Nốt phổi | BENIGN | MALIGN | Bệnh nhân | Tỷ lệ nốt |
|---|---:|---:|---:|---:|---:|
| Train/development | 786 | 503 | 283 | 435 | 79.07% |
| Test | 208 | 147 | 61 | 116 | 20.93% |
| Tổng | 994 | 650 | 344 | 551 | 100% |

Tập test được giữ riêng bằng fold đầu tiên của `StratifiedGroupKFold` năm fold với seed 42. Các nốt của cùng một bệnh nhân nằm hoàn toàn trong một partition, kể cả khi bệnh nhân có các nốt mang nhãn khác nhau.

Trong 786 nốt development, sử dụng **5-fold cross-validation theo bệnh nhân** để chọn mô hình và siêu tham số. Điền giá trị thiếu, encoding, scaling, lọc đặc trưng hằng và lựa chọn đặc trưng đều nằm trong Pipeline, được fit riêng trong phần training của từng fold. Không có bước lựa chọn đặc trưng hoặc tuning sử dụng dữ liệu test.

Hai thuật toán được so sánh là Logistic Regression và Random Forest, có điều chỉnh class weight. DummyClassifier theo tỷ lệ lớp được dùng làm baseline. Tiêu chí lựa chọn là ROC-AUC trung bình trên CV. Sau khi chọn, mô hình được fit lại trên toàn bộ 786 nốt development và đánh giá một lần trên 208 nốt test tại ngưỡng cố định 0.5.

## 5. Kết quả cross-validation

Bảng dưới trình bày cấu hình tốt nhất của từng cặp nhóm đặc trưng và thuật toán. Giá trị sau dấu ± là **độ lệch chuẩn giữa năm fold**, không phải khoảng tin cậy.

| Nhóm đặc trưng | Mô hình | ROC-AUC CV, trung bình ± SD | Average precision CV | Balanced accuracy CV |
|---|---|---:|---:|---:|
| Tabular | **Random Forest** | **0.9420 ± 0.0140** | 0.9196 | 0.8689 |
| Tabular | Logistic Regression | 0.9367 ± 0.0182 | 0.8980 | 0.8770 |
| Kết hợp | **Logistic Regression** | **0.9344 ± 0.0155** | 0.9055 | 0.8729 |
| Kết hợp | Random Forest | 0.9304 ± 0.0154 | 0.9004 | 0.8564 |
| Radiomics | Random Forest | 0.9271 ± 0.0176 | 0.8982 | 0.8453 |
| Radiomics | Logistic Regression | 0.9250 ± 0.0177 | 0.8900 | 0.8421 |
| Baseline | Dummy prior | 0.5000 ± 0.0000 | 0.3624 | 0.5000 |

**Random Forest tabular được chọn để đánh giá cuối cùng** vì đạt ROC-AUC CV cao nhất. Cấu hình gồm 200 cây, `max_depth = 6`, `min_samples_leaf = 2`, `class_weight = balanced`, seed 42 và giữ 100% đặc trưng sau bước lọc phương sai.

Đối với yêu cầu sử dụng cả VOI và tabular, mô hình kết hợp tốt nhất là Logistic Regression với `C = 0.1` và giữ 50% đặc trưng sau lọc phương sai bằng xếp hạng ANOVA. Mô hình này đã được fit trên toàn bộ development và lưu riêng. Điểm CV cho thấy việc thêm radiomics chưa cải thiện ROC-AUC so với tabular trong phép so sánh hiện tại; kết quả này chưa chứng minh sự khác biệt có ý nghĩa thống kê hoặc áp dụng cho các bộ dữ liệu khác.

Các điểm CV được dùng để chọn cấu hình nên có thể lạc quan. Kết quả test dưới đây là đánh giá của mô hình đã được chọn trước khi xem test.

## 6. Kết quả trên tập test

Các chỉ số sau thuộc **Random Forest tabular**, không phải mô hình kết hợp. Ngoài baseline, các mô hình còn lại không được đánh giá riêng trên tập test trong thí nghiệm hiện tại.

| Chỉ số | Random Forest tabular | Dummy baseline |
|---|---:|---:|
| ROC-AUC | **0.9597** | 0.5000 |
| Average precision | **0.8689** | 0.2933 |
| Accuracy | **90.87%** | 70.67% |
| Balanced accuracy | **90.66%** | 50.00% |
| Precision cho MALIGN | **80.88%** | 0.00% |
| Recall / sensitivity cho MALIGN | **90.16%** | 0.00% |
| Specificity cho BENIGN | **91.16%** | 100.00% |
| F1 cho MALIGN | **85.27%** | 0.00% |
| Brier score, thấp hơn tốt hơn | **0.0734** | 0.2117 |

Ngưỡng phân loại là **0.5**. Khoảng tin cậy bootstrap theo bệnh nhân 95% cho ROC-AUC là **[0.9308; 0.9848]**, từ 1.000 lần lấy mẫu. Khoảng này phản ánh biến thiên lấy mẫu bệnh nhân trong tập test, không bao gồm toàn bộ bất định từ training, tuning hoặc thay đổi quần thể. Tỷ lệ MALIGN trong test là 29.33%; đây cũng là mức tham chiếu của average precision với baseline prior.

Confusion matrix, hàng là nhãn thực tế và cột là nhãn dự đoán:

| Nhãn thực tế | Dự đoán BENIGN | Dự đoán MALIGN | Tổng |
|---|---:|---:|---:|
| BENIGN | **134** | 13 | 147 |
| MALIGN | 6 | **55** | 61 |
| Tổng | 140 | 68 | 208 |

Mô hình dự đoán đúng **189/208 nốt**. Có 13 nốt mang nhãn BENIGN được dự đoán MALIGN và 6 nốt mang nhãn MALIGN được dự đoán BENIGN. Các con số này mô tả sai khác với nhãn của bộ dữ liệu, không phải số ca ung thư được xác nhận hoặc bị bỏ sót trong thực hành lâm sàng.

Biểu đồ ROC, precision–recall và confusion matrix có trong output của [notebook 02](02_malign_benign_modeling.ipynb). Khi chạy lại, ảnh được lưu vào `artifacts/modeling/test_evaluation.png`.

## 7. Đặc trưng và lưu mô hình

Các đặc trưng có impurity importance lớn nhất trong Random Forest tabular là:

| Đặc trưng | Importance |
|---|---:|
| Đường kính `diameter_mm` | 0.5114 |
| `Subtlety_value` | 0.1855 |
| `Spiculation_value` | 0.0819 |
| `Lobulation_value` | 0.0625 |

Importance mô tả cách mô hình đã fit sử dụng đặc trưng; không chứng minh quan hệ nhân quả hoặc độ ổn định của biomarker. Impurity importance có thể ưu tiên biến liên tục, nên không được diễn giải thành phần trăm đóng góp vào nguy cơ ung thư.

Bốn model bundle đã được lưu trong `artifacts/modeling/`:

- `malign_benign_model.joblib`: mô hình được chọn cuối cùng, Random Forest tabular.
- `tabular_best_model.joblib`: mô hình tabular tốt nhất theo CV.
- `radiomics_best_model.joblib`: mô hình radiomics tốt nhất theo CV.
- `combined_best_model.joblib`: mô hình kết hợp tốt nhất theo CV, Logistic Regression.

Các bundle chứa Pipeline, schema đầu vào, ngưỡng phân loại, cấu hình extraction và phiên bản thư viện. Tất cả vẫn chỉ được fit trên development. Việc nạp lại model trong một tiến trình Python khác đã tái tạo đúng toàn bộ 208 dự đoán test. Pipeline inference của model kết hợp cũng đã được kiểm tra: đọc lại VOI/mask và tabular cho cùng một mẫu tạo ra score khớp với dự đoán từ bảng đặc trưng đã lưu. Kiểm tra này xác nhận tính nhất quán của inference, không phải một đánh giá test độc lập bổ sung.

## 8. Diễn giải và giới hạn

Pipeline đã hoàn thành trích radiomics, ghép dữ liệu, huấn luyện và đánh giá phân loại BENIGN/MALIGN. Trong các cấu hình đã thử, Random Forest tabular có ROC-AUC CV cao nhất và cho kết quả tốt trên tập test giữ riêng theo bệnh nhân. Mô hình kết hợp VOI/radiomics và tabular đã được triển khai đầy đủ nhưng chưa vượt tabular về ROC-AUC CV.

Các mô tả tabular và nhãn đều đến từ đánh giá hình ảnh của bác sĩ. Vì vậy, mô hình có thể học cách tái tạo đánh giá của người đọc; kết quả chưa chứng minh khả năng chẩn đoán ung thư trên nhãn mô bệnh học. Model kết hợp cần các mô tả tabular đó ở thời điểm dự đoán, còn model radiomics cần VOI và mask của nốt phổi.

Các giới hạn khác gồm chưa xác nhận cường độ HU, chất lượng và tính ổn định của segmentation, một bộ dữ liệu, một lần chia holdout và phạm vi siêu tham số có giới hạn. Cần validation độc lập để đánh giá khả năng tổng quát hóa. Nếu điều chỉnh pipeline hoặc chọn ngưỡng sau khi xem kết quả test, cần một tập test độc lập mới để báo cáo hiệu năng cuối cùng.

## 9. Nguồn kết quả

- `artifacts/radiomics/extraction_run.json`: số nốt, số đặc trưng, cấu hình và phiên bản extraction.
- `artifacts/radiomics/qc.csv`: kiểm tra từng cặp VOI/mask.
- `artifacts/modeling/split_assignments.csv`: phân chia bệnh nhân và nốt phổi.
- `artifacts/modeling/cv_summary.csv`: bảng so sánh CV.
- `artifacts/modeling/test_metrics.json`: chỉ số và khoảng bootstrap trên test.
- `artifacts/modeling/test_predictions.csv`: toàn bộ 208 dự đoán test.
- `artifacts/modeling/selected_feature_importance.csv`: importance của mô hình được chọn.
- Hai notebook lưu đầy đủ code và outputs của lượt chạy.

## 10. Thử nghiệm bổ sung feature engineering

Sau thí nghiệm ban đầu, notebook [03_feature_engineering_ablation.ipynb](03_feature_engineering_ablation.ipynb) đã thử thêm tám đặc trưng tự tạo ở EDA: `log_volume`, `intensity_difference`, `inside_median`, `inside_iqr`, ba kích thước mask và `mask_fill_ratio`. Thí nghiệm dùng đúng 786 nốt development và năm fold đã lưu. Tập test và các model baseline giữ nguyên.

Sau tuning với cùng ngân sách tìm kiếm, ROC-AUC CV của nhánh kết hợp tăng nhẹ từ **0.9344 lên 0.9357**, nhưng F1 CV giảm từ **0.8312 xuống 0.8264**. Nhánh radiomics tăng nhẹ cả hai chỉ số; nhánh tabular có điểm tốt nhất trước khi thêm feature cao hơn sau khi thêm. Các kết quả này chưa cho thấy cải thiện nhất quán hoặc chứng minh ý nghĩa thống kê. Chúng là số đo development CV, không thay thế các chỉ số test tại mục 6.

Bảng đối chiếu trước/sau và so sánh giữ nguyên cấu hình nằm trong [FEATURE_ENGINEERING_RESULTS.md](FEATURE_ENGINEERING_RESULTS.md). Các model thêm feature đã được lưu riêng trong `artifacts/feature_engineering/`, vẫn chỉ fit trên development và chưa được đánh giá trên test độc lập.
