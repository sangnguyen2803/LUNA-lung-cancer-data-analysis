# Lung Nodule Classification with PyRadiomics and Machine Learning

**Project report**

**Date:** 3 November 2026

## Summary

This project studied BENIGN/MALIGN classification using the supplied LUNA lung nodule data. It compared radiomics features extracted from 3D images with metadata describing nodule size and appearance. The analysis included data checks, feature extraction, model training, patient-based evaluation and several improvement experiments.

PyRadiomics extracted 107 features from all 994 available image-mask pairs. The original selected model was a Random Forest using nine metadata inputs. On the original test set of 208 nodules from 116 patients, it achieved **90.87% accuracy, 0.9597 ROC-AUC and 85.27% F1 for MALIGN**. Additional features, threshold tuning, stronger classifiers and ensembles did not provide reliable evidence of an accuracy improvement. The original Random Forest remains the recommended model for this project.

The target labels come from radiologist ratings. They are **not pathology-confirmed cancer diagnoses**. The results describe agreement with the supplied labels.

## Contents

- [Aim and scope](#aim-and-scope)
- [Dataset and target](#dataset-and-target)
- [EDA](#eda)
- [PyRadiomics feature extraction](#pyradiomics-feature-extraction)
- [Modeling](#modeling)
- [Original test results](#original-test-results)
- [Extra features](#extra-features)
- [F1 tuning](#f1-tuning)
- [Accuracy tuning](#accuracy-tuning)
- [Stronger models and ensembles](#stronger-models-and-ensembles)
- [Native categorical boosting](#native-categorical-boosting)
- [Implementation and reproducibility](#implementation-and-reproducibility)
- [Limitations and future work](#limitations-and-future-work)
- [Conclusion](#conclusion)

## Aim and scope

The main question was whether image-derived radiomics features could improve prediction compared with nodule metadata alone. A second question was whether feature engineering, different classifiers or threshold selection could improve the initial results.

Each observation was one nodule. The inputs were a 3D volume of interest (VOI), its supplied segmentation mask and the available nodule descriptors. The work covered classification of already identified nodules. Automatic detection in full CT scans and training a segmentation model were outside the completed work.

## Dataset and target

The two main tables were `MetadatabyNoduleMaxVoting.xlsx` and `MetadatabyAnnotation.xlsx`. Records were matched using `patient_id`, `nodule_id` and `seriesuid` together. This kept annotations from different scans separate.

| Item | Count |
|---|---:|
| Nodules in the main metadata table | 996 |
| Annotation records | 3,674 |
| Patients represented in the modeling cohort | 551 |
| CT series listed in the metadata | 552 |
| Available VOI images | 994 |
| Available VOI masks | 996 |
| Complete pairs used for modeling | 994 |
| BENIGN nodules in the modeling cohort | 650 |
| MALIGN nodules in the modeling cohort | 344 |

The complete metadata table contained 650 BENIGN and 346 MALIGN nodules. Two MALIGN VOI images were missing: `LIDC-IDRI-1002_R_1.nii.gz` and `LIDC-IDRI-1002_R_2.nii.gz`. Their masks and metadata were present. These two nodules were excluded from every feature-set comparison, including metadata-only models, so all groups used the same cohort.

The target was the supplied `Diagnosis`, encoded as BENIGN = 0 and MALIGN = 1. The dataset assigns MALIGN when at least two annotation records have `Malignancy_value > 3`. The label audit reproduced this rule for all 996 metadata rows without mismatches. An aggregated malignancy score alone could not reconstruct every label.

Several records can describe the same nodule. Repeated annotation keys were therefore expected and were not removed as duplicate observations. The annotation table also contained one extra nodule/scan group for another series of patient `LIDC-IDRI-0332`.

## EDA

The exploratory analysis checked table quality, label distributions, image geometry and mask properties. Both metadata tables had no missing values or fully duplicated rows, and the main table had unique nodule keys.

The labels were unequal: approximately 65% of the complete modeling cohort was BENIGN. This made a majority-class baseline necessary and meant that accuracy needed to be considered together with MALIGN precision, recall and F1.

Nodule size showed a strong group difference in the full metadata table:

| Diameter | BENIGN | MALIGN |
|---|---:|---:|
| Mean | 6.25 mm | 13.28 mm |
| Median | 5.74 mm | 12.46 mm |

The groups still overlapped. These differences did not define a diagnostic size cutoff. Other patterns involved spiculation, lobulation, subtlety and calcification, but no single descriptor fully separated the labels.

The image checks found 994 usable pairs. There were 972 LPS-oriented and 22 RAS-oriented VOIs. Thirty-five masks had disconnected components and were flagged for review. Reversed X/Y coordinate bounds in 22 nodules were consistent with orientation-related coordinate ordering checks; these findings were reviewed rather than treated as automatic reasons for exclusion.

Annotation scores also disagreed: 136 nodules had identical scores across their records, while 142 had a score range of at least three points. This showed that a final binary label can hide uncertainty between readers.

Only three full CT images were available locally. The complete classification experiments therefore used the supplied VOI crops rather than a full-CT detection workflow.

## PyRadiomics feature extraction

Images and masks were checked for matching 3D dimensions, spacing, origin and direction. Image values had to be finite, and masks had to be binary and contain a usable region. All labeled components were retained; disconnected masks were flagged rather than reduced to their largest component.

The extraction protocol used:

- LPS image orientation;
- 1 x 1 x 1 mm resampled voxel spacing;
- linear interpolation for images and nearest-neighbor interpolation for masks;
- mask label 1;
- Original images only;
- a fixed bin width of 25 stored intensity units.

Linear interpolation was used consistently because several two-slice crops failed with B-spline interpolation in the tested environment. The same crops produced finite features with linear interpolation.

Intensity calibration to Hounsfield units was not verified. No HU clipping or intensity normalization was assumed. Settings were fixed before model training and recorded in [radiomics_params.yaml](radiomics_params.yaml).

PyRadiomics extracted seven feature families:

| Family | Features |
|---|---:|
| Shape | 14 |
| First-order | 18 |
| GLCM | 24 |
| GLRLM | 16 |
| GLSZM | 16 |
| GLDM | 14 |
| NGTDM | 5 |
| **Total** | **107** |

Extraction succeeded for **994 of 994 available pairs**. The resulting radiomics table contained no missing feature values. Missing images remained documented in the exclusion report.

## Modeling

### Input groups and leakage prevention

Three input groups were compared:

| Group | Raw inputs |
|---|---|
| Metadata | Diameter and eight visual descriptors: 9 inputs |
| Radiomics | Image-derived features: 107 inputs |
| Combined | Metadata and radiomics: 116 inputs |

The eight visual descriptors were calcification, internal structure, lobulation, margin, sphericity, spiculation, subtlety and texture. Calcification and internal structure were treated as nominal categories. The other six scores were treated as ordered numeric inputs. Age, sex and clinical history were not available.

Diagnosis columns, malignancy scores, malignancy-vote summaries and `len_mal_details` were excluded from predictors. Patient IDs, scan IDs, nodule IDs, filenames and absolute coordinates were also excluded. Patient IDs were used only for grouping.

A metadata model requires the visual descriptors at inference time. A combined model requires those descriptors as well as the VOI and mask. The metadata inputs and target labels both reflect readers' assessments, which limits clinical interpretation.

### Patient-based split

A fixed `StratifiedGroupKFold` split with seed 42 reserved approximately 20% of patients for the original test. Every nodule from a patient stayed in the same partition.

| Partition | Nodules | Patients | BENIGN | MALIGN |
|---|---:|---:|---:|---:|
| Development | 786 | 435 | 503 | 283 |
| Original test | 208 | 116 | 147 | 61 |
| Total | 994 | 551 | 650 | 344 |

Five patient-grouped development folds were used for initial model selection. Median imputation, categorical imputation and one-hot encoding, numeric scaling, constant-feature removal and ANOVA feature selection were fitted inside each training fold.

Logistic Regression and Random Forest were compared using balanced class weights. Each input group used six Logistic Regression settings and eight Random Forest settings. The initial selection criterion was mean development ROC-AUC. A prior-probability DummyClassifier provided the baseline.

### Initial development results

These scores were used for model selection and can be optimistic. They are not independent test results.

| Input group | Classifier | Mean CV ROC-AUC | SD |
|---|---|---:|---:|
| Metadata | Random Forest | **0.9420** | 0.0140 |
| Metadata | Logistic Regression | 0.9367 | 0.0182 |
| Radiomics | Random Forest | 0.9271 | 0.0176 |
| Radiomics | Logistic Regression | 0.9250 | 0.0177 |
| Combined | Logistic Regression | 0.9344 | 0.0155 |
| Combined | Random Forest | 0.9304 | 0.0154 |

The selected model was the metadata Random Forest with 200 trees, maximum depth 6, minimum leaf size 2, balanced class weights and all nonconstant encoded features. Its random seed was 42. In the initial experiment, this selected model and the dummy baseline were evaluated on the original test after model selection.

## Original test results

The following results belong to the selected metadata Random Forest, at threshold 0.5. They are not test scores for the radiomics-only or combined models.

| Metric | Random Forest | Dummy baseline |
|---|---:|---:|
| ROC-AUC | **0.9597** | 0.5000 |
| Average precision | **0.8689** | 0.2933 |
| Accuracy | **90.87%** | 70.67% |
| Balanced accuracy | **90.66%** | 50.00% |
| MALIGN precision | **80.88%** | 0.00% |
| MALIGN recall | **90.16%** | 0.00% |
| BENIGN specificity | **91.16%** | 100.00% |
| MALIGN F1 | **85.27%** | 0.00% |
| Brier score | **0.0734** | 0.2117 |

The patient-bootstrap 95% interval for test ROC-AUC was **[0.9308, 0.9848]**, based on 1,000 resamples. This interval reflects sampling variation among test patients, not all uncertainty from training or tuning. Average precision should be read alongside the test MALIGN prevalence of 29.33%.

The confusion matrix was:

| Actual label | Predicted BENIGN | Predicted MALIGN |
|---|---:|---:|
| BENIGN | 134 | 13 |
| MALIGN | 6 | 55 |

The model matched 189 of 208 dataset labels. It produced 13 false-positive and six false-negative predictions relative to those labels. These are not counts of pathology-confirmed cancers detected or missed.

The largest Random Forest impurity importances were diameter (0.5114), subtlety (0.1855), spiculation (0.0819) and lobulation (0.0625). These describe the fitted model. They do not establish causal effects or percentages of cancer risk, and impurity importance can favor continuous variables.

## Extra features

Eight native-image measurements were added: log volume, inside-minus-outside mean intensity, inside median, inside IQR, three mask-box sizes in mm and mask fill ratio. These measurements used the supplied masks and native voxel spacing.

The experiment reused the 786 development nodules and the original folds. It first kept baseline model settings fixed, then retuned with the same search budget. The table shows the retuned comparisons. F1 was measured on development folds at the default decision rule.

| Input group | AUC before | AUC after | F1 before | F1 after |
|---|---:|---:|---:|---:|
| Metadata | 0.9420 | 0.9399 | 0.8281 | 0.8250 |
| Radiomics | 0.9271 | 0.9290 | 0.8037 | 0.8059 |
| Combined | 0.9344 | 0.9357 | 0.8312 | 0.8264 |

Radiomics improved slightly, while the combined model gained AUC but lost F1. The fixed-model comparison also showed mixed changes. These results did not demonstrate a consistent improvement. The original test was not reevaluated for these augmented models.

## F1 tuning

A nested experiment tried 28 combined-input configurations using Logistic Regression, Random Forest and histogram gradient boosting. One input variant added three engineered features. Five outer patient folds evaluated the procedure, and three inner patient folds selected models and thresholds by MALIGN F1. Candidate thresholds ranged from 0.10 to 0.90 in steps of 0.025.

| Procedure | Mean outer F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Fixed original combined model, threshold 0.5 | 0.8312 | 0.7958 | 0.8720 | 0.9344 |
| Fixed combined model, inner threshold tuning | 0.8155 | 0.7895 | 0.8473 | 0.9344 |
| F1-selected model, threshold 0.5 | 0.8106 | 0.7937 | 0.8358 | 0.9238 |
| F1-selected model and threshold | 0.8027 | 0.7903 | 0.8220 | 0.9238 |
| Nested combined ROC-AUC reference, threshold 0.5 | 0.8278 | 0.7925 | 0.8685 | 0.9299 |

The fixed original settings had already been selected on development data, so that row was a descriptive reference. The nested ROC-AUC reference repeated the original combined search inside each outer training set. Against that reference, F1 model-and-threshold selection reduced mean F1 by 2.51 percentage points.

The search spaces differed, so the comparison did not isolate the scoring rule alone. No new independent test result was produced for the F1 candidate.

## Accuracy tuning

A separate 37-configuration experiment considered Logistic Regression, Random Forest, Extra Trees and histogram gradient boosting. It compared metadata and selected combined inputs, class weights and thresholds. Models and thresholds were chosen inside the inner folds.

| Procedure | Mean outer accuracy | MALIGN F1 | MALIGN recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Fixed original metadata model | 87.48% | 0.8281 | 85.23% | 0.9420 |
| Nested metadata ROC-AUC selection | **88.59%** | **0.8473** | **87.87%** | **0.9413** |
| Nested accuracy and threshold selection | 86.08% | 0.8015 | 78.47% | 0.9313 |

Accuracy-focused selection reduced mean accuracy by 2.50 percentage points relative to the nested metadata ROC-AUC reference. Recall and F1 also fell. The fixed original row used previously selected settings and was not a fresh nested model-selection estimate.

These are development CV scores. They must not be compared directly with the original test accuracy of 90.87%. The accuracy candidate was saved separately and was not evaluated on the original test.

## Stronger models and ensembles

The next experiment expanded the search to 52 candidates. It included the original 14 metadata configurations, CatBoost, calibrated SVM, soft voting and stacking. SVM calibration and stacking used patient-grouped folds within their training sets. Selection used inner ROC-AUC, and the classification threshold stayed at 0.5.

| Search procedure | Mean outer accuracy | ROC-AUC | MALIGN F1 | MALIGN recall |
|---|---:|---:|---:|---:|
| Original 14-setting search | **88.59%** | 0.9413 | **0.8473** | **87.87%** |
| Expanded 52-setting search | 87.28% | **0.9436** | 0.8284 | 85.44% |

The expanded search raised mean AUC slightly but reduced accuracy, F1 and recall. Its final development-selected candidate was a voting model combining metadata Logistic Regression, metadata Random Forest and combined-input Logistic Regression, with weights `[1, 2, 1]`.

### Comparison using the original test again

The voting configuration and model file were frozen before its test predictions were calculated. However, the original test outcomes had already been viewed during earlier work. This was therefore an **exploratory reused-test comparison**, not a new independent test. The voting candidate was not retuned from this comparison.

| Model | Accuracy | ROC-AUC | MALIGN precision | MALIGN recall | MALIGN F1 |
|---|---:|---:|---:|---:|---:|
| Original Random Forest | **90.87%** | 0.9597 | **80.88%** | 90.16% | **85.27%** |
| Frozen voting candidate | 90.38% | **0.9602** | 79.71% | 90.16% | 84.62% |

The voting model made one additional false-positive prediction, while false negatives stayed at six. Its accuracy change was -0.48 percentage points. The paired patient-bootstrap 95% interval was **[-2.26, +1.01] percentage points**. This interval covers cohort sampling variation, not all training or adaptive-search uncertainty.

The small AUC increase did not establish a reliable overall improvement. The original model, split assignments and original test prediction files were preserved.

## Native categorical boosting

A final completed experiment used LightGBM with calcification and internal structure kept as categorical columns. It compared eight new settings with the original 14 metadata settings. A native CatBoost runtime pilot had been stopped; it was not included as a completed performance comparison.

The LightGBM candidates used 300 trees, learning rate 0.05, depths 3 or 4, regularization values 3 or 10, and balanced or unweighted classes. Imputation and category conversion stayed inside the fitted pipeline. Integer and floating-point category codes were converted consistently.

The selection rule was fixed before examining this experiment's outer results. A new candidate had to improve mean inner accuracy by at least 0.5 percentage points, lose at most two recall points and lose at most 0.005 AUC relative to the inner ROC-AUC reference. Otherwise, the reference was retained. Thresholds stayed at 0.5.

| Selection procedure | Mean outer accuracy | ROC-AUC | MALIGN F1 | MALIGN recall |
|---|---:|---:|---:|---:|
| Original search | 88.59% | 0.9413 | 0.8473 | 87.87% |
| Guarded search with LightGBM candidates | 88.59% | 0.9413 | 0.8473 | 87.87% |

All five outer folds retained the reference. The final development search also returned the original metadata Random Forest settings. The identical scores describe the fallback procedure; they are not standalone LightGBM scores.

The native experiment changed both the learner and category handling, so it did not isolate the effect of encoding alone. It was designed after earlier experiments and test outcomes had been seen, making its development results exploratory. Its saved candidate was not evaluated again on the original test.

## Implementation and reproducibility

The project keeps the analysis in five notebooks:

| Notebook | Purpose |
|---|---|
| [luna_explore_dataset.ipynb](luna_explore_dataset.ipynb) | Data exploration and image-mask checks |
| [01_voi_radiomics.ipynb](01_voi_radiomics.ipynb) | Preprocessing and PyRadiomics extraction |
| [02_malign_benign_modeling.ipynb](02_malign_benign_modeling.ipynb) | Initial model comparison, original test and inference |
| [03_feature_engineering_ablation.ipynb](03_feature_engineering_ablation.ipynb) | Extra-feature comparison |
| [04_f1_optimization.ipynb](04_f1_optimization.ipynb) | F1, accuracy, ensemble and native-category experiments |

The tested environment uses Python 3.9, NumPy 1.26.4, pandas 2.2.3, SimpleITK 2.4.1, PyRadiomics 3.1.0, scikit-learn 1.6.1, CatBoost 1.2.10 and LightGBM 4.6.0. Dependencies are pinned in [requirements-radiomics.txt](requirements-radiomics.txt).

From PowerShell in the project directory, the existing radiomics environment can run the complete workflow and checks with:

```powershell
.\.venv-radiomics\Scripts\python.exe run.py --include-f1
.\.venv-radiomics\Scripts\python.exe test_pipeline.py -v
```

`--smoke` runs a reduced check on 60 selected pairs in a separate output directory. `--improve` runs notebook 04 using previously generated outputs from notebooks 01-03. Environment setup is described in [README.md](README.md).

Generated features, audits, folds, predictions, metrics and model bundles are stored under `artifacts/`, which is ignored by Git. Extraction caches use input hashes, settings and package versions. The selected original model is `artifacts/modeling/malign_benign_model.joblib`; later candidates are stored separately.

Verification completed with **25 of 25 automated tests passing**. The full-data experiments and smoke workflow had no cell errors. Checks covered image/mask rejection, extraction on thin crops, worker recovery, patient separation, out-of-fold tuning, saved thresholds, grouped calibration/stacking, category types and training-only imputation.

Reloading the voting model in a fresh Python process reproduced all 208 saved scores. Prediction from a raw VOI, mask and metadata also matched its stored-feature score. The native fallback reproduced the original model on development data. Hash checks confirmed that original baseline and test files were unchanged. These checks establish implementation consistency, not clinical validity.

The main numerical sources for this report are `artifacts/modeling/test_metrics.json`, `cv_summary.csv`, `split_assignments.csv`, and the experiment summaries under `artifacts/feature_engineering/`, `f1_optimization/`, `accuracy_optimization/`, `model_improvement/` and `native_boosting/`. Dataset definitions come from the supplied Excel tables and [LUNA16_Dataset_Description.pdf](LUNA16_Dataset_Description.pdf).

## Limitations and future work

1. **Assessment-based labels.** The models learn the supplied radiologist-vote labels. Performance cannot be interpreted as pathology-confirmed cancer diagnosis.
2. **Shared source of labels and descriptors.** Metadata predictors and target labels reflect assessments of the same images. A model may reproduce readers' rating behavior.
3. **Limited validation.** The original result used one patient-held-out split from one dataset. Later experiments repeatedly reused development data, and one voting comparison reused the original test. New independent patients are needed for a final claim about improved generalization.
4. **Image and mask uncertainty.** HU calibration was not verified, some masks had separate components, and geometry checks did not establish the quality or stability of every segmentation.
5. **Uncalibrated clinical meaning.** Classification scores, especially from class-weighted models, should not be interpreted as clinical cancer probabilities.
6. **Restricted inputs.** Age, sex and clinical history were unavailable. Image-based predictions depended on supplied masks, and metadata models required the reader descriptors.

Further work should validate intensity units and segmentation quality, examine label disagreement, and evaluate a frozen pipeline on a new patient cohort. Any additional threshold or model tuning should be completed using development data before that new evaluation.

## Conclusion

The project completed a reproducible pipeline for exploring the data, extracting radiomics, training models and evaluating BENIGN/MALIGN classification by patient. PyRadiomics produced 107 features for all 994 available pairs.

The original metadata Random Forest remains the recommended model, with 90.87% accuracy, 0.9597 ROC-AUC and 85.27% MALIGN F1 on the original test. Radiomics, extra features, threshold tuning, stronger classifiers and ensembles did not provide reliable evidence for an accuracy improvement in the recorded experiments. The study supports retaining the original model while seeking independent validation of any future changes.
