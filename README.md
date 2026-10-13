# LUNA Lung Cancer Data Analysis

## Dataset folders

Place the dataset next to the notebook, using this structure:

```text
LUNA-lung-cancer-data-analysis/
├── CT/
│   └── CT/
│       ├── image/          # Full 3D CT scans (.nii.gz)
│       └── nodule_mask/    # Nodule masks in the full CT
├── VOIs/
│   └── VOIs/
│       ├── image/          # Small 3D images around nodules (.nii.gz)
│       └── nodule_mask/    # Masks for those small images
├── MetadatabyNoduleMaxVoting.xlsx
├── MetadatabyAnnotation.xlsx
├── LUNA16_Dataset_Description.pdf
└── luna_explore_dataset.ipynb
```

Keep both folder levels: `CT/CT/` and `VOIs/VOIs/`. Start Jupyter from the project folder so the notebook can find the files.

## What each file contains

| File or folder | Contents |
|---|---|
| `CT/CT/image/` | Full CT scans. One scan can contain several nodules. |
| `CT/CT/nodule_mask/` | Masks marking individual nodules in a full CT. |
| `VOIs/VOIs/image/` | A VOI is a 3D crop around one nodule, including nearby tissue. |
| `VOIs/VOIs/nodule_mask/` | A matching mask: 1 marks the nodule, 0 marks the rest. |
| `MetadatabyNoduleMaxVoting.xlsx` | One row per listed nodule, with its size, location, feature scores and final label. Use this as the main table. |
| `MetadatabyAnnotation.xlsx` | Individual score records. Several rows can describe the same nodule. Match them using `patient_id`, `nodule_id` and `seriesuid`. |
| `LUNA16_Dataset_Description.pdf` | Dataset details, feature codes and label rules. |
| `luna_explore_dataset.ipynb` | Data checks, plots and image examples. |

For example, `LIDC-IDRI-0001_R_1.nii.gz` is a 3D VOI for nodule 1 of patient `LIDC-IDRI-0001`. Its mask has the same filename in the mask folder. The slices are inside this file; they do not have separate nodule IDs.
