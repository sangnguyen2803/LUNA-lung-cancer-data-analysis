# LUNA Lung Cancer Data Analysis

Explore nodule metadata, 3D CT images and masks in a Jupyter notebook.

## Setup

Install Python and Git, then run these commands in **PowerShell**:

```powershell
git clone https://github.com/sangnguyen2803/LUNA-lung-cancer-data-analysis.git
cd LUNA-lung-cancer-data-analysis

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib scipy nibabel openpyxl jupyterlab ipykernel

.\.venv\Scripts\python.exe -m jupyterlab
```

Open `luna_explore_dataset.ipynb`, select the Python kernel, then choose **Run > Run All Cells**.

## Data

The dataset files are included in this repo. Keep `CT/`, `VOIs/` and both Excel files in the project root. Start JupyterLab from that folder so the notebook can find them.
