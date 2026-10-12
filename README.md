# Heart Disease Risk Prediction Using Big Data Technologies and Machine Learning

An end-to-end Big Data Analytics project that combines Apache Kafka, Hadoop Distributed File System (HDFS), and machine learning to demonstrate clinical data ingestion, distributed storage, predictive modeling, and interactive heart disease risk assessment.

The project uses the UCI Cleveland Heart Disease dataset to compare Logistic Regression and Random Forest classifiers and presents their predictions and evaluation results through a Streamlit dashboard.

> **Disclaimer:** This is an academic project developed for educational and analytical purposes. Its predictions are not medical diagnoses and must not be used as a substitute for professional medical advice.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Running the Data Pipeline](#running-the-data-pipeline)
- [Viewing Data in HDFS](#viewing-data-in-hdfs)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Heart disease risk assessment involves evaluating multiple clinical attributes and identifying patterns associated with the presence of heart disease.

This project demonstrates how Big Data technologies and machine learning can be combined to build a data processing and prediction workflow.

The system integrates:

- **Apache Kafka** for streaming clinical records.
- **Hadoop HDFS** for distributed storage of ingested records.
- **Python and scikit-learn** for data preprocessing, model training, and evaluation.
- **Streamlit** for an interactive dashboard that enables users to explore the dataset, assess sample clinical profiles, and review model performance.

The project focuses on demonstrating data engineering, machine learning, and application integration within a reproducible local development environment.

## Key Features

- Kafka-based streaming of clinical records.
- Kafka consumer for validating and processing incoming records.
- Hadoop HDFS storage using JSON Lines (JSONL).
- WebHDFS integration for accessing stored records.
- Missing-value handling and feature preprocessing using scikit-learn pipelines.
- Comparison of Logistic Regression and Random Forest classifiers.
- Stratified train-test splitting and repeated stratified cross-validation.
- Model evaluation using ROC-AUC, accuracy, precision, recall, specificity, and F1-score.
- Interactive Streamlit dashboard with six pages:
  - Dashboard
  - Risk Assessment
  - Analysis Report
  - Model Evaluation
  - Dataset Explorer
  - About / Methodology
- TXT and JSON report downloads for the latest assessment, as implemented in the dashboard.

---

## System Architecture

The project uses the following data processing and prediction workflow:

```text
             UCI Cleveland Heart Disease Dataset
                            |
                            v
                  Dataset Preparation
                            |
                            v
                    Kafka Producer
                            |
                            v
                  Kafka Topic
             heart-disease-records
                            |
                            v
                    Kafka Consumer
                            |
                            v
                     Hadoop HDFS
                            |
                            v
                Data Preprocessing
                            |
                            v
                 Machine Learning
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Logistic Regression    Random Forest
                 |                     |
                 +----------+----------+
                            |
                            v
                 Model Evaluation
                            |
                            v
                  Streamlit Dashboard
                            |
                            v
              Interactive Risk Assessment
```

### Architecture Components

| Component | Purpose |
|---|---|
| Dataset preparation | Loads and prepares the Cleveland dataset for analysis. |
| Apache Kafka | Streams clinical records through a Kafka topic. |
| Kafka consumer | Receives and validates streamed records. |
| Hadoop HDFS | Stores ingested records in JSONL format. |
| WebHDFS | Provides HTTP-based access to HDFS data. |
| Preprocessing pipeline | Handles missing values, scales numerical features, and encodes categorical features. |
| Machine learning | Trains and evaluates Logistic Regression and Random Forest classifiers. |
| Streamlit dashboard | Provides dataset exploration, risk assessment, model evaluation, and report generation. |

The infrastructure is configured as a local development cluster using Docker Compose. The Kafka ingestion workflow can be demonstrated using the supplied producer and consumer scripts.

---

## Technology Stack

| Technology | Usage |
|---|---|
| Python | Application development and data processing |
| pandas | Dataset manipulation and analysis |
| NumPy | Numerical computation |
| scikit-learn | Preprocessing, classification, and model evaluation |
| Apache Kafka | Event streaming using KRaft mode |
| Hadoop HDFS | Distributed file storage |
| WebHDFS | HTTP-based HDFS access |
| Streamlit | Interactive web dashboard |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization and loading |
| Docker | Containerized infrastructure |
| Docker Compose | Multi-container service orchestration |
| Jupyter Notebook | Exploratory data analysis and model experimentation |

---

## Dataset

The project uses the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.

**Dataset source:** [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Dataset Summary

| Property | Description |
|---|---|
| Dataset | UCI Cleveland Heart Disease |
| Number of records | 303 |
| Input features | 13 |
| Target variable | `target` |
| Classification type | Binary classification |
| Missing values | Present in `ca` and `thal` |

### Input Features

The model uses the following 13 attributes:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex indicator as encoded in the dataset |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels visualized by fluoroscopy |
| `thal` | Thalassemia-related dataset category |

### Target Variable

The original target is converted into a binary classification target:

- `0` — No heart disease indicated by the dataset's binary target definition.
- `1` — Heart disease indicated by the dataset's binary target definition.

The processed dataset contains 164 records in class 0 and 139 records in class 1.

Missing values in `ca` and `thal` are retained during dataset preparation and handled within the machine learning preprocessing pipeline.

---

## Machine Learning Models

Two classification algorithms are trained and compared.

### 1. Logistic Regression

Logistic Regression estimates the probability of a binary outcome using a weighted combination of input features. It provides a useful baseline for binary classification.

### 2. Random Forest

Random Forest combines predictions from multiple decision trees. It can model nonlinear relationships and interactions between input features.

### Preprocessing and Evaluation

The existing machine learning workflow includes:

- Stratified 80:20 train-test split.
- 242 training records and 61 held-out test records.
- Missing-value imputation.
- Standardization of numerical features.
- One-hot encoding of categorical features.
- Repeated stratified five-fold cross-validation, repeated five times.
- Evaluation on the held-out test set.

Preprocessing is encapsulated within scikit-learn pipelines to keep transformations consistent between training and prediction and to prevent test data from influencing the fitted preprocessing steps.

---

## Results

The following metrics are from the existing held-out test evaluation.

### Held-Out Test Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 90.16% | 85.25% |
| ROC-AUC | 0.9697 | 0.9470 |
| Precision | 86.67% | 82.76% |
| Recall | 92.86% | 85.71% |
| Specificity | 87.88% | 84.85% |
| F1-score | 0.8966 | 0.8421 |

### Cross-Validation ROC-AUC

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8956 | 0.0421 |
| Random Forest | 0.8936 | 0.0404 |

### Evaluation Metrics Explained

- **Accuracy:** Proportion of test records classified correctly.
- **ROC-AUC:** Measures how well a model distinguishes between the two target classes across classification thresholds.
- **Precision:** Proportion of positive predictions that are actually positive.
- **Recall:** Proportion of actual positive records correctly identified.
- **Specificity:** Proportion of actual negative records correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

Logistic Regression achieved higher held-out test scores across the reported metrics in this experiment. However, the test set is small, and the results should not be interpreted as evidence of clinical effectiveness.

The Streamlit dashboard provides visualizations and model evaluation information from the saved project results.

---

## Project Structure

```text
heart-disease-big-data/
├── data/
│   ├── raw/
│   └── processed/
│       └── heart_disease.csv
├── dashboard/
│   └── app.py
├── docker/
│   └── hadoop.env
├── hdfs/
│   ├── hdfs_writer.py
│   └── test_kafka_hdfs.py
├── kafka/
│   ├── config.py
│   ├── producer.py
│   ├── consumer.py
│   ├── hdfs_consumer.py
│   ├── demo_producer.py
│   ├── demo_consumer.py
│   └── test_streaming.py
├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── pipeline_metadata.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── preprocessing/
│   ├── prepare_dataset.py
│   └── pipeline.py
├── results/
│   ├── model_metrics.json
│   ├── metrics_comparison.csv
│   ├── test_predictions.csv
│   ├── roc_curve_comparison.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
├── docs/
│   ├── dataset_documentation.md
│   └── model_documentation.md
├── .streamlit/
│   └── config.toml
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** Trained `.joblib` model files are generated locally and excluded from version control. The dashboard requires the trained model artifacts to be available in the expected `models/` directory.

---

## Prerequisites

The following software is required to run the project locally:

- Windows 11.
- Python 3.10 or later.
- Git.
- Docker Desktop with the WSL2 backend enabled.
- Sufficient memory for the Kafka and Hadoop containers.

The commands below use Windows PowerShell.

## Installation and Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/pulkitsinha17/heart-disease-big-data.git
cd heart-disease-big-data
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, consult the Python documentation for virtual environment activation on Windows.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Big Data Infrastructure

Ensure Docker Desktop is running, then execute:

```powershell
docker compose up -d
```

Check service status:

```powershell
docker compose ps
```

The project uses these local endpoints:

| Service | Address |
|---|---|
| Kafka broker | `localhost:9092` |
| HDFS NameNode web interface | `http://localhost:9870` |
| HDFS DataNode web endpoint | `http://localhost:9864` |
| Streamlit dashboard | `http://localhost:8501` |

Kafka clients running inside the Docker network use `kafka:29092` where configured.

Allow the containers time to initialize before using the services.

---

## Running the Application

### Launch the Streamlit Dashboard

From the project root, with the virtual environment activated:

```powershell
streamlit run dashboard/app.py
```

Open the local address displayed by Streamlit, usually:

**http://localhost:8501**

The dashboard includes:

- **Dashboard:** Project overview and model metrics.
- **Risk Assessment:** Submit a clinical feature profile to obtain predictions from the available models.
- **Analysis Report:** View the latest assessment and download its report where supported.
- **Model Evaluation:** Compare model performance and inspect evaluation visualizations.
- **Dataset Explorer:** Explore the dataset and its attributes.
- **About / Methodology:** Review the project's methodology and limitations.

The model artifacts must be available locally for prediction functionality.

---

## Running the Data Pipeline

Ensure Docker services are running and the Python virtual environment is activated.

### 1. Stream Records Through Kafka

The primary Kafka topic is:

```text
heart-disease-records
```

The producer reads the processed dataset and publishes records to Kafka.

In one PowerShell terminal, run the consumer:

```powershell
python kafka/consumer.py --max-messages 303
```

In another terminal, run the producer:

```powershell
python kafka/producer.py
```

### 2. Store Kafka Records in HDFS

The HDFS ingestion workflow consumes Kafka records and writes them to HDFS.

In one terminal, start the consumer:

```powershell
python kafka/hdfs_consumer.py --max-messages 303
```

In another terminal, publish the records:

```powershell
python kafka/producer.py
```

The target HDFS file is:

```text
/heart-disease/raw/heart_disease_stream.jsonl
```

**Important:** These commands perform real ingestion and may append data or affect Kafka offsets. Before repeating a run, inspect the existing topic and HDFS output to avoid duplicate records. Do not delete topics or stored data simply to rerun the demonstration.

For additional integration-test instructions, inspect `hdfs/test_kafka_hdfs.py` and `kafka/test_streaming.py`.

---

## Viewing Data in HDFS

### List the Stored Files

```powershell
docker exec namenode hdfs dfs -ls -h /heart-disease/raw
```

### View the Beginning of the JSONL File

```powershell
docker exec namenode hdfs dfs -head /heart-disease/raw/heart_disease_stream.jsonl
```

### View the End of the JSONL File

```powershell
docker exec namenode hdfs dfs -tail /heart-disease/raw/heart_disease_stream.jsonl
```

### Open the HDFS Web Interface

Visit:

http://localhost:9870

The interface provides information about the NameNode and the HDFS filesystem.

---

## Limitations

- The dataset contains only 303 records from the Cleveland cohort and may not represent broader patient populations.
- The dataset is historical, and its attributes and target definitions may not reflect current clinical practice.
- The held-out test set contains only 61 records, so individual predictions and evaluation metrics are subject to uncertainty.
- High test ROC-AUC values do not establish clinical reliability or guarantee generalization to new populations.
- The local Docker setup demonstrates distributed-data technologies but is not a production-scale cluster.
- The dashboard's predictions are generated by trained machine learning models and should not be interpreted as medical diagnoses.

Clinical use would require independent validation on larger and more representative datasets, expert review, and appropriate governance.

---

## References

1. **UCI Machine Learning Repository — Heart Disease Dataset**  
   https://archive.ics.uci.edu/dataset/45/heart+disease

2. **Apache Kafka Documentation**  
   https://kafka.apache.org/documentation/

3. **Apache Hadoop Documentation**  
   https://hadoop.apache.org/docs/

4. **scikit-learn Documentation**  
   https://scikit-learn.org/stable/

5. **Streamlit Documentation**  
   https://docs.streamlit.io/

6. **Docker Documentation**  
   https://docs.docker.com/

---

## License

No license is specified in this repository unless a license file is provided. Please contact the repository owner for permission to reuse or distribute this project.
