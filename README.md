# Computer Vision Final Project - Multi Class Remote Sensing Image Segmentation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets
- Visit https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx
- Download the Potsdam & Vaihingen zip files
- extract them and gather the following parts:

**For Potsdam:**
- `2_Ortho_RGB.zip` 
- `5_Labels_all.zip`

**For Vaihingen:**
- `ISPRS_semantic_labeling_Vaihingen.zip` (images in folder: "top")
- `ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip`

Put the image files in the following folder structure:
```
data/
├── potsdam/
│   ├── images/
│   └── labels/
└── vaihingen/
    ├── images/
    └── labels/
```

## 📊 Dataset Information

### Land Cover Classes

Both datasets contain 6 semantic classes:

| Class ID | Name | Color | Description |
|----------|------|-------|-------------|
| 0 | Impervious surfaces | White | Roads, parking lots, etc. |
| 1 | Building | Blue | Residential/commercial buildings |
| 2 | Low vegetation | Cyan | Grass, small plants |
| 3 | Tree | Green | Trees and forests |
| 4 | Car | Yellow | Vehicles |
| 5 | Clutter/background | Red | Other objects |

### Dataset Splits

**Potsdam:**
- Train: Areas 2_10, 2_13, 2_14, 3_10, 3_11, 3_12, 3_13, 4_10, 4_12, 4_13, 4_14, 5_10, 5_12, 5_13, 5_14, 5_15, 6_8, 6_9, 6_10, 6_11, 6_12, 6_13, 6_15, 7_7, 7_9, 7_11, 2_11, 2_12, 3_14, 4_11, 7_13
- Test: Areas Areas 4_15, 5_11, 6_7, 6_11
- Validation: Areas 6_14, 7_8, 7_10, 7_12

**Vaihingen:**
- Train: Areas 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
- Test: Areas 31, 32, 33, 34
- Validation: Areas 35, 36, 37, 38