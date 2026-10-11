# Heart Disease Risk Prediction Using Kafka, Hadoop HDFS and Machine Learning

## 1. Problem Statement
Primary-care screening for cardiovascular risk currently relies on manual scoring rules that do not account for interactions between multiple risk factors.

## 2. Objective
Ingest patient clinical measurements via Kafka, store them on Hadoop HDFS, and train classification models that predict heart disease risk with ROC-AUC above 0.85 on a held-out test set.

---

## 3. Planned Architecture

The planned end-to-end architecture follows a big data streaming, storage, and machine learning pipeline:

```
[Patient Data Stream / UCI Cleveland Dataset]
                     │
                     ▼
        [Apache Kafka Producer]
                     │
              (Kafka Topic)
                     │
                     ▼
        [Apache Kafka Consumer]
                     │
                     ▼
         [Hadoop HDFS Storage]
       (/data/raw distributed storage)
                     │
                     ▼
    [Batch Preprocessing & Feature Pipeline]
       (Missing value imputation, scaling, encoding)
                     │
                     ▼
     [Machine Learning Models (scikit-learn)]
  (Model Training, Hyperparameter Tuning, ROC-AUC > 0.85)
                     │
                     ▼
         [Model Artifacts (joblib)]
                     │
                     ▼
     [Interactive Dashboard (Streamlit)]
       (Clinical Risk Prediction Interface)
```

### Architectural Components:
1. **Data Ingestion Layer**: Apache Kafka producer simulates streaming of patient clinical records from primary care screening checkpoints into a designated Kafka topic.
2. **Distributed Storage Layer**: A Kafka consumer ingests clinical events in real time and persists raw records into Hadoop Distributed File System (HDFS) for fault-tolerant, scalable storage.
3. **Data Preprocessing & Feature Engineering Layer**: Reads persisted records, handles missing indicators, standardizes numerical features, and encodes categorical health indicators.
4. **Machine Learning Pipeline**: Trains multiple candidate classification algorithms (e.g., Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines) to predict presence of heart disease, validating generalization on a held-out test set with a target ROC-AUC > 0.85.
5. **Presentation & Clinical Interface Layer**: Serializes high-performing models via `joblib` and exposes an interactive Streamlit dashboard for clinicians to input clinical measurements and obtain risk probabilities.

---

## 4. Planned Technology Stack

- **Programming Language**: Python
- **Streaming Ingestion**: Apache Kafka (`kafka-python`)
- **Distributed Big Data Storage**: Apache Hadoop HDFS (`hdfs`)
- **Exploratory Analysis & Prototyping**: Jupyter Notebook (`jupyter`, `notebook`)
- **Data Manipulation & Computation**: pandas, NumPy
- **Machine Learning**: scikit-learn
- **Data Visualization**: matplotlib, seaborn
- **Model Serialization**: joblib
- **Interactive UI / Dashboard**: Streamlit (planned for downstream phase)
- **Containerization**: Docker / Docker Compose (for Kafka, Zookeeper, and Hadoop services)

---

## 5. Planned Project Phases

- **Phase 1: Project Structuring & Environment Setup** *(Completed)*
  - Establish directory structure, dependencies (`requirements.txt`), and documentation.
- **Phase 2: Dataset Acquisition & Exploratory Data Analysis (EDA)**
  - Acquire the UCI Heart Disease (Cleveland) dataset.
  - Perform statistical analysis and feature correlation visualizations in Jupyter Notebooks.
- **Phase 3: Streaming Ingestion Pipeline with Apache Kafka**
  - Implement Kafka Producer to simulate continuous clinical patient measurements.
  - Implement Kafka Consumer to read from Kafka topics.
- **Phase 4: Distributed Storage on Hadoop HDFS**
  - Configure Hadoop HDFS storage.
  - Ingest raw stream data into HDFS directories.
- **Phase 5: Data Cleaning & Feature Engineering**
  - Implement preprocessing scripts for missing value handling, feature scaling, and categorical encoding.
- **Phase 6: Machine Learning Model Development & Validation**
  - Train classification models using scikit-learn.
  - Evaluate performance against the target threshold: **ROC-AUC > 0.85** on a held-out test set.
- **Phase 7: Model Serialization & Export**
  - Save trained pipeline and model artifacts using `joblib`.
- **Phase 8: Clinical Dashboard Development**
  - Build an interactive Streamlit application for primary-care screening predictions.
- **Phase 9: Containerization & Final Documentation**
  - Provide Docker Compose setups for Kafka and Hadoop, along with final reporting.

---

## 6. Directory Layout

```
heart-disease-big-data/
│
├── data/
│   ├── raw/                  # Raw dataset files and ingested Kafka batches
│   └── processed/            # Cleaned, transformed feature matrices
│
├── kafka/                    # Kafka producers, consumers, and topic configurations
│
├── hdfs/                     # HDFS client utilities and ingestion scripts
│
├── notebooks/                # Jupyter notebooks for EDA and experimental modeling
│
├── preprocessing/            # Data cleaning and feature engineering pipelines
│
├── models/                   # Serialized model binaries (.joblib) and training logic
│
├── results/                  # Evaluation metrics, ROC curves, and reports
│
├── dashboard/                # Streamlit web application code
│
├── docker/                   # Docker Compose and container setup files
│
├── docs/                     # Project documentation, architecture diagrams, and reports
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and roadmap
└── .gitignore                # Version control ignore rules
```
