# LUNA Lung Cancer Data Analysis

A notebook-based study of lung nodules: explore the data, extract image features with
PyRadiomics, and compare models for the supplied BENIGN / MALIGN labels.

## Notebooks

Run the notebooks from top to bottom in this order. The EDA notebook can run separately.

| Notebook | Main task |
|---|---|
| [luna_explore_dataset.ipynb](luna_explore_dataset.ipynb) | Explore tables, labels, CT images and nodule masks. |
| [01_voi_radiomics.ipynb](01_voi_radiomics.ipynb) | Check VOI/mask pairs and extract 107 PyRadiomics features. |
| [02_malign_benign_modeling.ipynb](02_malign_benign_modeling.ipynb) | Compare metadata, radiomics and combined models; evaluate and save the winner. |
| [03_feature_engineering_ablation.ipynb](03_feature_engineering_ablation.ipynb) | Check whether eight extra image features improve the models. |
| [04_f1_optimization.ipynb](04_f1_optimization.ipynb) | Try model and threshold selection for MALIGN F1 using nested cross-validation. |

All analysis code is in the notebooks. `scripts/execute_pipeline.py` is an optional
runner; `tests/` contains checks for image handling, patient splits and prediction.

## Data

Keep the following files and folders next to the notebooks:

```text
CT/CT/image/                    # Full 3D CT scans (.nii.gz)
CT/CT/nodule_mask/              # Masks in the full CT scans
VOIs/VOIs/image/                # 3D crops around individual nodules
VOIs/VOIs/nodule_mask/          # Matching masks: 1 = nodule, 0 = background
MetadatabyNoduleMaxVoting.xlsx  # One row per nodule: main metadata table
MetadatabyAnnotation.xlsx       # Individual radiologist score records
LUNA16_Dataset_Description.pdf
```

Keep both folder levels (`CT/CT/` and `VOIs/VOIs/`). A VOI and its mask share a
filename, for example `LIDC-IDRI-0001_R_1.nii.gz`. Each file is a 3D volume.
Join the two tables using `patient_id`, `nodule_id` and `seriesuid` together.

There are 996 metadata rows and 994 available VOI/mask pairs from 551 patients.
The two missing VOI images are recorded in the exclusion report.
Masks are supplied by the dataset; the notebooks do not detect or segment nodules.

## Setup and run

The tested Windows setup uses **Python 3.9**, **PyRadiomics 3.1.0** and
**NumPy 1.26.4**. Python 3.9 supports the PyRadiomics Windows wheel used here;
the existing Python 3.14 EDA environment is unsuitable for this setup.
Install the `pyradiomics` distribution, imported in Python as `radiomics`.

From PowerShell in the project directory:

```powershell
py -3.9 -m venv .venv-radiomics
.\.venv-radiomics\Scripts\python.exe -m pip install --only-binary pyradiomics -r requirements-radiomics.txt
.\.venv-radiomics\Scripts\python.exe -m ipykernel install --user --name luna-radiomics --display-name "LUNA Radiomics (Python 3.9)"
.\.venv-radiomics\Scripts\python.exe -m jupyterlab
```

Alternatively, create the environment with `uv venv .venv-radiomics --python 3.9 --seed`
and install with `uv pip install --python .venv-radiomics\Scripts\python.exe --only-binary pyradiomics -r requirements-radiomics.txt`.
In Jupyter, select **LUNA Radiomics (Python 3.9)** and run notebooks 01 through 04.

To run the core extraction and modeling notebooks automatically:

```powershell
.\.venv-radiomics\Scripts\python.exe scripts\execute_pipeline.py
```

Optional runner flags:

| Flag | What runs |
|---|---|
| `--smoke` | A quick check on 60 pairs with smaller searches; outputs go to `artifacts/smoke/`. |
| `--feature-engineering` | Only notebook 03; requires outputs from 01 and 02. |
| `--include-feature-engineering` | Notebooks 01 through 03. |
| `--f1` | Only notebook 04; requires outputs from 01 through 03. |
| `--include-f1` | All four notebooks. |

Combine `--smoke` with an experiment flag to use smoke outputs. Run the required
earlier smoke notebooks first. The runner uses its own Python interpreter for Jupyter.

To run the checks:

```powershell
.\.venv-radiomics\Scripts\python.exe -m unittest discover -s tests -v
```

## Method and results

- PyRadiomics extracts seven feature families from Original images. Settings are
  in [radiomics_params.yaml](radiomics_params.yaml): LPS orientation in notebook 01,
  1 mm resampling, linear image interpolation, nearest-neighbor mask interpolation
  and bin width 25. Linear interpolation also works with the four thin crops that
  failed with B-spline in this environment. Intensity units have not been verified
  as HU; no HU clipping or intensity normalization is applied.
- Models compare 9 metadata inputs, 107 radiomics inputs, and their combination.
  Metadata consists of diameter and eight visual descriptors; age, sex and clinical
  history are unavailable. Diagnosis, malignancy scores/votes and patient IDs are
  excluded from predictors. Combined predictions require those visual descriptors.
- Patients are split into development and test groups before fitting preprocessing.
  All nodules from one patient stay together. Imputation, encoding, scaling and
  feature selection are fitted inside development cross-validation folds.
- The selected **Random Forest using metadata** reached **ROC-AUC 0.9597**,
  **accuracy 90.87%** and **MALIGN F1 85.27%** on 208 test nodules from 116 patients,
  at threshold 0.5. These are the selected metadata model's test scores.
- Extra features and F1 tuning did not show consistent improvement. Their scores
  come from development cross-validation; the saved candidates have no new
  independent test evaluation.

Read [EDA_results.md](EDA_results.md), [MODEL_RESULTS.md](MODEL_RESULTS.md),
[FEATURE_ENGINEERING_RESULTS.md](FEATURE_ENGINEERING_RESULTS.md) and
[F1_OPTIMIZATION_RESULTS.md](F1_OPTIMIZATION_RESULTS.md) for the full results.
Plots are also saved in the notebook outputs.

## Generated files and limits

Notebooks create `artifacts/` for feature tables, quality checks, splits, metrics,
plots and saved models. This folder, virtual environments and caches are ignored
by Git. Rerun the notebooks to recreate them. Successful extractions are cached by
input content, settings and package versions; rerun notebook 01 after changing
inputs or extraction settings. Notebook 02 checks the saved feature hashes.

The target is the supplied radiologist-vote-derived `Diagnosis` (BENIGN = 0,
MALIGN = 1), **not pathology-confirmed cancer**. Results describe these labels on
one dataset. Models need independent validation before clinical claims. Further
tuning after viewing the original test requires a new independent test set.
