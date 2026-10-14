# LUNA Dataset: EDA Results

This summary follows Sections 1–6 of [the notebook](luna_explore_dataset.ipynb). Mean means average; median means the middle value.

The analysis is a good starting point: it covers table quality, missing files, labels, feature patterns and 3D image-mask checks. It also explains score disagreement and image spacing. Sections 7 and 8 explain which tasks the dataset can support and what to do next. No model performance has been measured yet.

## 1. Load data

### 1.1 Tables and image files

The main table has **996 nodules**. The annotation table has **3,674 score records**.

One nodule can have several score records. Match the tables using `patient_id`, `nodule_id` and `seriesuid` together. An annotation row is a set of scores, not an image slice.

### 1.2 CT and VOI examples

The example CT has shape **512 × 512 × 133**. Its nodule VOI has shape **53 × 64 × 11**.

Both files are 3D. A CT covers a larger area and can contain several nodules. A VOI is a crop around one nodule. Each displayed picture is one slice of that 3D file.

### 1.3 Voxels in this project

| Measurement | Result |
|---|---:|
| Voxels per VOI | 3,698–352,000 |
| Total voxels in 994 VOIs | 20,405,061 |
| Median spacing along X, Y, Z | About 0.70 × 0.70 × 1.25 mm |
| Largest VOI shape | 80 × 80 × 55 |

The **352,000 voxels** count the whole crop, including tissue around the nodule. They are not all nodule voxels.

For `LIDC-IDRI-0001_R_1.nii.gz`, the VOI has **37,312 voxels**, but its mask marks only **6,547** as nodule voxels. With spacing **0.703125 × 0.703125 × 2.5 mm**, its mask volume is **8,091.84 mm³**.

Nodule volume = mask voxel count × spacing X × spacing Y × spacing Z. Voxel counts alone cannot compare physical size when spacing differs.

## 2. Check the data

### 2.1 Missing values and repeated rows

- Both tables have **0 missing values** and **0 fully duplicated rows**.
- The nodule table has **0 duplicated nodule keys**.
- The annotation table has **2,677 repeated keys**, because several score records describe the same nodule and scan.
- There are **551 patients** and **552 CT series**. `LIDC-IDRI-0332` has two series; the other patients have one each.

Repeated annotation keys are expected. Removing them as duplicates would lose score records. Missing table values and missing image files are separate checks.

### 2.2 Score codes, sizes and boxes

All checked codes are allowed, diameters are positive, and numeric values have no NaN or infinity.

**22 nodules** have reversed X/Y bounds. The **86 flagged annotation rows** describe the same 22 nodules. All 22 VOIs have RAS voxel-axis directions. After converting the box corners back to voxel coordinates, their Low/High order is correct.

This suggests that the Low/High names follow voxel order while the saved coordinates in mm run in the opposite direction. It is not evidence of 22 invalid boxes, and RAS does not always imply reversed bounds. The export code is unavailable, so the exact creation process is not confirmed.

**2 nodule rows and 7 annotation rows** have equal Z bounds and still need review before cropping. Keep the flagged rows; do not remove them based on these checks alone.

### 2.3 Available and missing files

| Data | Count |
|---|---:|
| Local full CT images | 3 |
| Local masks in full CT scans | 6 |
| VOI images | 994 |
| VOI masks | 996 |
| Missing VOI images | 2 |
| Missing VOI masks | 0 |

The missing images are `LIDC-IDRI-1002_R_1.nii.gz` and `LIDC-IDRI-1002_R_2.nii.gz`. Both have the label `Malign`; their metadata and masks are available.

Use **996 nodules** for table analysis and **994 pairs** for image analysis. The three local full CT scans are only a small part of the 552 series listed in the metadata.

### 2.4 Nodules and score records

**317 of 551 patients** have one nodule. Patients have **1–9 nodules** each. In the annotation table, **323 groups** have three records and **668** have four; six groups have five to seven.

There are **997 nodule/CT groups** in the annotation table, compared with 996 in the main table. Section 4 explains the extra group. These record counts do not tell us how many different doctors were involved, because there is no doctor ID.

## 3. Nodule labels and features

### 3.1 Labels and malignancy scores

| Label | Nodules | Percentage |
|---|---:|---:|
| Benign | 650 | 65.3% |
| Malign | 346 | 34.7% |

The groups are unequal: about two thirds are `Benign`. Also, **91 Malign nodules** have a saved `Malignancy_value` of 1, 2 or 3. The saved score alone cannot rebuild `Diagnosis`, because the label uses individual score votes.

These labels describe the dataset's voting rule, not confirmed tissue-test results.

### 3.2 Nodule sizes

| Diameter (mm) | Benign | Malign |
|---|---:|---:|
| Mean | 6.25 | 13.28 |
| Median | 5.74 | 12.46 |

Overall, diameters range from **3.25 to 32.27 mm**, with a median of **6.77 mm**. The IQR rule flags **66 large nodules above 18.26 mm: 63 Malign and 3 Benign**.

The Malign group is larger on average, but the groups overlap. A large nodule is not automatically Malign, and an IQR flag is a reason to review a row rather than delete it.

### 3.3 Correlations

The heatmap uses **Spearman correlation** for all **22 numeric columns**. The ranking table shows the ten strongest positive pairs and ten strongest negative pairs, without self-pairs or repeated pairs.

| Pair | Spearman |
|---|---:|
| `coordZ` and `bboxHighZ` | 0.999 |
| `diameter_mm` and `Diagnosis_value` | 0.694 |
| `diameter_mm` and `Malignancy_value` | 0.527 |
| `Spiculation_value` and `Malignancy_value` | 0.437 |
| `Calcification_value` and `Margin_value` | -0.360 |
| `Malignancy_value` and `Margin_value` | -0.301 |

A positive value means the two values tend to rise together. A negative value means one tends to fall as the other rises. Values near zero show little consistent link in their ranks.

Nine of the ten top positive pairs are center and box coordinates: they describe the same location. The strongest negative pair is `Calcification_value` with `Margin_value`, but its link is much weaker than those coordinate links.

IDs are identifiers, and some numeric columns are category codes. A larger category code does not always mean more of a feature. Correlation does not prove cause or show how well a model will work.

### 3.4 Features by diagnosis

Percentages are calculated separately for **650 Benign** and **346 Malign** nodules.
For example, 94.6% means about 95 out of every 100 Benign nodules have the stated feature. It is not 94.6% of the whole dataset.

| Feature and code | Meaning | Benign (%) | Malign (%) |
|---|---|---:|---:|
| `Spiculation_value = 1` | No spiky edges | 94.6 | 57.5 |
| `Lobulation_value = 1` | No lobed edge (no rounded bumps) | 86.8 | 50.0 |
| `Texture_value = 5` | Solid | 84.9 | 78.3 |
| `Calcification_value = 6` | No calcification | 82.3 | 98.8 |
| `Margin_value = 5` | Sharp, clear edge | 44.8 | 23.7 |
| `Subtlety_value = 5` | Obvious / easy to see | 21.7 | 70.5 |
| `InternalStructure_value = 1` | Soft tissue | 100.0 | 99.7 |

**What this shows**

- The Benign group more often has no spikes or lobes, and a sharp edge.
- The Malign group more often has an obvious appearance and no calcification.
- Solid texture and soft-tissue structure are common in both groups. `Sphericity` scores also overlap.

The groups share many values, so a single feature cannot fully separate the labels. Category codes such as `Calcification_value = 6` are names represented by numbers, not amounts.

### 3.5 Diameter distribution by diagnosis

The added histogram uses the same bins and percentages within each group. This lets us compare the distributions despite different group sizes.

| Diagnosis | Mean (mm) | Median (mm) | Skew | At or below 10 mm (%) |
|---|---:|---:|---:|---:|
| Benign | 6.25 | 5.74 | 3.33 | 94.77 |
| Malign | 13.28 | 12.46 | 0.66 | 32.37 |

Both distributions have a tail toward large diameters. The Benign group has stronger skew: most nodules are small, but a few large ones pull its mean up. The Malign group is larger on average and covers a wider range of diameters.

At 10 mm, the within-group percentages are very different, but the distributions still overlap. This is an example for understanding the data, not a chosen prediction cutoff. Large sizes are not automatically data errors.

## 4. Annotation and label analysis

### 4.1 Labels and record counts

The rule is: label a nodule `Malign` when **at least two records have `Malignancy_value > 3`**; otherwise label it `Benign`.

All **996 nodules** have matching score groups. There are **0 label mismatches** and **0 mismatches with `len_mal_details`**.

The tables agree with the stated rule. This checks how the labels were built, not whether each nodule is truly cancer.

### 4.2 Extra score group

One extra group contains **four records** for **LIDC-IDRI-0332, nodule 1**, in another CT series. It is absent from the main nodule table.

This explains the 997 groups versus 996 listed nodules. Including `seriesuid` when matching records keeps different scans apart.

### 4.3 Score differences

**136 nodules (13.7%)** have identical scores across records. **142 (14.3%)** have a gap of at least three points between the lowest and highest scores.

For example, `LIDC-IDRI-0141`, nodule 1 has scores **1, 2, 5 and 4**. Two are above 3, so its label is `Malign` despite the disagreement. One final label hides these score differences.

## 5. Image and mask analysis

### 5.1 Read images and masks

All **994 available image-mask pairs** pass the checks used for image analysis. Each pair matches in array shape and spatial position. All available images contain finite values.

This lets us compare values inside and outside each mask. The two masks without images can still be checked as masks, but cannot provide image-value measurements.

### 5.2 Image sizes, spacing and masks

VOIs have **2–62 slices**, and their spacing differs. There are **972 LPS** and **22 RAS** images: their stored axis directions differ. The previews turn them to the same directions.

All **996 masks** are nonempty, contain only 0 and 1, and have no NaN or infinity. **35 masks** contain more than one connected part. Parts count as connected when voxels touch at a face, edge or corner.

The median mask volume is **235.91 mm³**, while the mean is **1,068.58 mm³**. A few large nodules raise the mean. For `LIDC-IDRI-0179_R_9`, only **73.4%** of mask voxels belong to the largest part, so it deserves a closer look.

Matching shape and position does not prove every mask outline is correct. Review separate parts before removing them.

### 5.3 View nodules and masks

The notebook shows one example from each label group and one mask with separate parts. It displays the slice containing the largest mask area, with a colored mask overlay.

`LIDC-IDRI-0069_R_1` has **two parts in 3D**. One slice may not show both. The examples help us understand the files; they do not check every mask.

### 5.4 Compare volume and image values

This comparison uses **650 Benign** and **344 Malign** image-mask pairs. Each nodule contributes one set of measurements.

| Mean measurement | Benign | Malign |
|---|---:|---:|
| `volume_mm3` | 281.94 | 2,541.27 |
| `inside_mean` | -303.39 | -265.87 |
| `inside_std` | 282.25 | 253.65 |
| `outside_mean` | -655.61 | -663.16 |

The Malign group has larger mask volumes on average. Its average value inside the mask is also higher, meaning less negative. The groups still overlap, so these measurements do not give a clear label cutoff.

`inside_std` measures how much image values vary inside a nodule. `outside_mean` describes nearby tissue inside the VOI but outside the mask; its group averages are close. These are stored image values: their HU units have not been verified in the notebook.

## 6. Feature engineering

### 6.1 Create new features

The notebook adds eight features from the images and masks:

| New feature | Meaning |
|---|---|
| `log_volume` | Log of 1 + mask volume; compress large volumes while keeping their size order. |
| `intensity_difference` | Mean inside the nodule minus mean outside it, within the VOI. |
| `inside_median` | Middle image value inside the nodule; less affected by extreme values than the mean. |
| `inside_iqr` | Percentile 75 minus percentile 25; measures the spread of values inside the nodule. |
| `mask_size_x_mm`, `mask_size_y_mm`, `mask_size_z_mm` | Width of the smallest box around all mask voxels, along each axis, in mm. |
| `mask_fill_ratio` | Nodule voxel count divided by voxel count in that small box. |

Mask sizes include the full width of the end voxels and use voxel spacing. Masks are turned to the same axis directions before measuring. These are nodule mask sizes, not whole VOI sizes. All mask parts are included.

These features need a nodule mask when used for predictions. The image files are unchanged.

### 6.2 Compare the new features

| Mean measurement | Benign | Malign |
|---|---:|---:|
| `log_volume` | 5.10 | 7.22 |
| `intensity_difference` | 352.22 | 397.29 |
| `inside_median` | -329.63 | -223.23 |
| `inside_iqr` | 431.88 | 394.68 |
| `mask_size_x_mm` | 8.06 | 19.22 |
| `mask_size_y_mm` | 8.07 | 19.11 |
| `mask_size_z_mm` | 7.19 | 15.51 |
| `mask_fill_ratio` | 0.445 | 0.356 |

The Malign group has larger mask sizes along all three axes on average. Its `log_volume` is also higher. The log reduces large size differences; it does not scale all features to the same range.

Both groups have positive `intensity_difference`: image values inside nodules are higher than those in nearby tissue on average. The Malign group has a higher mean `inside_median` but a lower mean `inside_iqr`. These describe different things: the median is the middle value, while IQR measures spread. Image values have not been confirmed as HU.

A mean `mask_fill_ratio` of 0.445 means the average nodule occupies about 44.5% of its tight box. The Malign group's average is about 35.6%. This is not the fraction of the whole VOI and does not directly measure roundness. Separate mask parts can also affect the box and its fill ratio.

These are group differences, not proof that the new features improve predictions.

### 6.3 Prepare a feature table

`X` contains **994 rows and eleven features**, with **0 missing values** and **0 non-finite values**:

- Original measurements: `diameter_mm`, `inside_mean`, `inside_std`.
- Added features: `log_volume`, `intensity_difference`, `inside_median`, `inside_iqr`, `mask_size_x_mm`, `mask_size_y_mm`, `mask_size_z_mm`, `mask_fill_ratio`.

`y` stores the label: **650 Benign (0)** and **344 Malign (1)**. Patient IDs are kept separately for a later split. The two missing-image nodules are outside this table.

IDs, `Diagnosis`, `Malignancy_value` and voting details are not inputs in `X`. Raw volume is kept for analysis, but this feature table uses its log version. Some features describe similar information; choose which to keep using training data only.

## 7. What did we learn?

The EDA found useful patterns that help us choose features and next tasks:

| Finding | What it tells us |
|---|---|
| Mean diameter is 13.28 mm for Malign vs 6.25 mm for Benign. Mean mask volume is 2,541.27 vs 281.94 mm3. | Size is a useful candidate feature for prediction, although the groups overlap. |
| Higher `Spiculation` and `Lobulation` scores are more common in the Malign group. | Shape-related scores may add information beyond size. Their value for prediction still needs testing. |
| Most top positive correlations are between center and box coordinates. | They repeat location information, rather than showing independent cancer-related patterns. |
| Only 136 nodules have identical scores across records; 142 have a score gap of at least 3. | A final label hides disagreement. Prediction will learn score-based labels. |
| All 994 available image-mask pairs align, but 35 masks have separate parts. | VOI segmentation experiments are possible after preprocessing and mask review. |

**Next tasks:** feature prediction and 3D image classification need a patient split and prepared inputs. The current feature table also needs masks when predicting. Full CT detection needs more scans; only three are available locally.

## 8. Next plan

These steps are planned, not completed:

1. Review missing images, flagged boxes and masks.
2. Split patients into training, validation and test groups.
3. Check HU units and prepare consistent image/mask inputs. Fit scaling on training data only.
4. Build feature prediction and VOI segmentation baselines. Check F1/recall for prediction and Dice/IoU for segmentation.
5. Use validation data to choose models, then evaluate on the test group and inspect errors.
