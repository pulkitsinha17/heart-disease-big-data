# Dataset Documentation: UCI Heart Disease (Cleveland)

## 1. Overview and Source
This project utilizes the benchmark **Heart Disease Dataset** hosted by the **UC Irvine Machine Learning Repository (UCI)**.
- **Repository Link**: [UCI Heart Disease Dataset (ID: 45)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Source URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Metadata Documentation**: `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` (`heart-disease.names`)

The full repository contains 4 distinct databases (Cleveland Clinic Foundation, Hungarian Institute of Cardiology Budapest, V.A. Medical Center Long Beach, and University Hospital Zurich). In accordance with clinical literature and the project specification, this project utilizes the **processed Cleveland database**, which is the gold standard benchmark for coronary artery disease risk prediction experiments.

### Citation & Principal Investigators
- **Cleveland Clinic Foundation**: Robert Detrano, M.D., Ph.D.
- **Key Reference**: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. **American Journal of Cardiology**, 64(5), 304–310.
- **Donor**: David W. Aha (July 1988)

---

## 2. Dataset Dimensions & Schema
- **Total Records (Patients)**: 303
- **Total Features**: 13 input clinical attributes + 1 binary target attribute
- **Raw Format**: Comma-separated values without headers, with missing entries indicated by `'?'`.

---

## 3. Feature Dictionary

| Attribute | Name in Dataset | Type | Clinical Description & Values |
| :--- | :--- | :--- | :--- |
| **Age** | `age` | Continuous (`float64`) | Age of patient in years (Range observed: 29.0 – 77.0). |
| **Sex** | `sex` | Categorical / Binary (`float64`) | Biological sex of the patient:<br>• `1.0`: Male<br>• `0.0`: Female |
| **Chest Pain Type** | `cp` | Categorical (`float64`) | Type of chest pain reported:<br>• `1.0`: Typical angina<br>• `2.0`: Atypical angina<br>• `3.0`: Non-anginal pain<br>• `4.0`: Asymptomatic |
| **Resting Blood Pressure** | `trestbps` | Continuous (`float64`) | Resting blood pressure in mm Hg measured upon admission to the hospital (Range observed: 94.0 – 200.0). |
| **Serum Cholesterol** | `chol` | Continuous (`float64`) | Serum cholesterol measurement in mg/dl (Range observed: 126.0 – 564.0). |
| **Fasting Blood Sugar** | `fbs` | Binary (`float64`) | Fasting blood sugar > 120 mg/dl indicator:<br>• `1.0`: True (> 120 mg/dl, suggestive of diabetes/hyperglycemia)<br>• `0.0`: False (<= 120 mg/dl) |
| **Resting ECG** | `restecg` | Categorical (`float64`) | Resting electrocardiographic results:<br>• `0.0`: Normal<br>• `1.0`: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)<br>• `2.0`: Showing probable or definite left ventricular hypertrophy by Estes' criteria |
| **Maximum Heart Rate** | `thalach` | Continuous (`float64`) | Maximum heart rate achieved during maximum treadmill exercise stress test (Range observed: 71.0 – 202.0). |
| **Exercise-Induced Angina** | `exang` | Binary (`float64`) | Exercise-induced angina pectoris:<br>• `1.0`: Yes<br>• `0.0`: No |
| **ST Depression (Oldpeak)** | `oldpeak` | Continuous (`float64`) | ST depression induced by exercise relative to rest on ECG (Range observed: 0.0 – 6.2 mm). |
| **Slope of Peak Exercise ST** | `slope` | Categorical (`float64`) | The slope of the peak exercise ST segment:<br>• `1.0`: Upsloping<br>• `2.0`: Flat<br>• `3.0`: Downsloping |
| **Number of Major Vessels** | `ca` | Discrete / Count (`float64`) | Number of major coronary vessels (0–3) colored by fluoroscopy. **Contains 4 missing values**. |
| **Thalassemia Status** | `thal` | Categorical (`float64`) | Thallium-201 myocardial perfusion scintigraphy status:<br>• `3.0`: Normal perfusion<br>• `6.0`: Fixed defect (non-reversible, prior infarction)<br>• `7.0`: Reversible defect (ischemia). **Contains 2 missing values**. |
| **Heart Disease Risk** | `target` | Binary (`int64`) | Presence of coronary artery disease (angiographic disease status):<br>• `0`: Absence (< 50% diameter narrowing in all major coronary vessels)<br>• `1`: Presence (> 50% diameter narrowing in at least 1 major vessel) |

---

## 4. Transformations Applied in Phase 2

1. **Header Mapping**: The raw UCI CSV is headerless; column names were mapped in standard order according to the documentation in `heart-disease.names`.
2. **Missing-Value Parsing**: The literal placeholder `'?'` was parsed into `np.nan` values.
3. **Target Binarization**:
   - The raw Cleveland dataset contains diagnosis grades `num` ranging from 0 to 4 (0 = no disease, 1 = mild, 2 = moderate, 3 = severe, 4 = critical).
   - In accordance with standard machine learning benchmark practice and the project objective:
     $$\text{target} = \begin{cases} 0 & \text{if } \text{num} = 0 \\ 1 & \text{if } \text{num} \ge 1 \end{cases}$$
   - Column `num` was transformed into integer `target` (`0` and `1`), and the original `num` column was dropped.
4. **No Row Deletions & No Imputation Yet**:
   - All 303 patient records were preserved.
   - Missing entries in `ca` (4 rows) and `thal` (2 rows) are retained as `NaN` to avoid data leakage or premature bias before downstream pipeline modeling.
   - No feature normalization or scaling has been applied yet.

---

## 5. Missing-Value Audit

| Feature | Missing Count | Missing Percentage | Action in Phase 2 |
| :--- | :--- | :--- | :--- |
| `ca` | 4 | 1.32% | Preserved as `NaN` (to be addressed in Phase 5 feature pipeline) |
| `thal` | 2 | 0.66% | Preserved as `NaN` (to be addressed in Phase 5 feature pipeline) |
| All other 12 features | 0 | 0.00% | None needed |
| **Total** | **6 entries** across 6 rows | **1.98% of rows** | Retained |

---

## 6. Class Distribution

| Target Value | Diagnosis Meaning | Row Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | Absence of Heart Disease (< 50% narrowing) | 164 | 54.13% |
| **1** | Presence of Heart Disease (> 50% narrowing) | 139 | 45.87% |
| **Total** | | **303** | **100.00%** |

The class distribution is well-balanced (approximately 54% negative vs 46% positive), avoiding severe imbalance issues that would require synthetic oversampling (SMOTE).

---

## 7. Storage Artifacts
- **Raw Download**: [`data/raw/processed.cleveland.data`](file:///c:/Users/pulki/OneDrive/Desktop/heart-disease-big-data/data/raw/processed.cleveland.data)
- **Raw Metadata**: [`data/raw/heart-disease.names`](file:///c:/Users/pulki/OneDrive/Desktop/heart-disease-big-data/data/raw/heart-disease.names)
- **Prepared Cleaned CSV**: [`data/processed/heart_disease.csv`](file:///c:/Users/pulki/OneDrive/Desktop/heart-disease-big-data/data/processed/heart_disease.csv)
- **Preparation Pipeline Script**: [`preprocessing/prepare_dataset.py`](file:///c:/Users/pulki/OneDrive/Desktop/heart-disease-big-data/preprocessing/prepare_dataset.py)
