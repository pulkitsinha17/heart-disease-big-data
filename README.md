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

## 4. Technology Stack

- **Programming Language**: Python 3.10+
- **Streaming Ingestion**: Apache Kafka (`apache/kafka:latest` running in KRaft mode; Python client: `kafka-python`)
- **Distributed Big Data Storage**: Apache Hadoop HDFS (`apache/hadoop:3`; Python client: `hdfs`)
- **Exploratory Analysis & Prototyping**: Jupyter Notebook (`jupyter`, `notebook`)
- **Data Manipulation & Computation**: pandas, NumPy
- **Machine Learning**: scikit-learn
- **Data Visualization**: matplotlib, seaborn
- **Model Serialization**: joblib
- **Interactive UI / Dashboard**: Streamlit (planned for downstream phase)
- **Containerization**: Docker & Docker Compose v2 (WSL2 backend)

---

## 5. Big Data Infrastructure (Docker Compose)

The repository provides a single-node development big data cluster managed via `docker-compose.yml`:

| Service | Container Name | Image | Internal Ports | Host Exposed Port | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kafka Broker** | `kafka` | `apache/kafka:latest` | 29092, 9093 | `localhost:9092` | Event streaming broker in KRaft mode (no ZooKeeper) |
| **HDFS NameNode**| `namenode` | `apache/hadoop:3` | 9870, 9000 | `localhost:9870`, `localhost:9000` | HDFS master metadata node & WebHDFS UI |
| **HDFS DataNode**| `datanode` | `apache/hadoop:3` | 9864, 9866, 9867 | `localhost:9864` | HDFS worker block storage node |

### Network & Connection Endpoints
- **Kafka Host Endpoint (for Python apps on Windows)**: `localhost:9092`
- **Kafka Container-to-Container Endpoint**: `kafka:29092`
- **HDFS NameNode Web UI & WebHDFS**: [http://localhost:9870](http://localhost:9870)
- **HDFS NameNode IPC / RPC**: `hdfs://localhost:9000` (external) / `hdfs://namenode:9000` (internal)
- **HDFS DataNode Web UI**: [http://localhost:9864](http://localhost:9864)

### Persistent Volumes
- `heart_disease_kafka_data`: Persists Kafka topic logs (`/var/lib/kafka/data`).
- `heart_disease_hadoop_namenode`: Persists HDFS FSImage and NameNode metadata (`/data/namenode`).
- `heart_disease_hadoop_datanode`: Persists HDFS block storage replicas (`/data/datanode`).

---

## 6. How to Run the Services

### Prerequisites
- Windows 11 with WSL2 enabled.
- Docker Desktop installed and running with the WSL2 backend.
- Git and Python virtual environment (refer to `requirements.txt`).

### 1. Start Services
From the project root directory, run:
```bash
docker compose up -d
```
Docker will download the images, initialize the network and volumes, automatically format the HDFS NameNode on first run, and start all containers.

### 2. Verify Service Status & Health
Check container state and healthcheck status:
```bash
docker compose ps
```
Both `kafka` and `namenode` report `(healthy)` once ready.

To view live container logs:
```bash
docker compose logs -f
```

### 3. Test Kafka Connectivity
Create an explicit test topic:
```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic heart-disease-test --partitions 1 --replication-factor 1
```

List existing topics:
```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

Delete the test topic:
```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic heart-disease-test
```

### 4. Test HDFS Connectivity
Check the HDFS cluster report (verify 1 Live DataNode is connected):
```bash
docker exec namenode hdfs dfsadmin -report
```

Create, list, and remove a directory in HDFS:
```bash
docker exec namenode hdfs dfs -mkdir -p /test
docker exec namenode hdfs dfs -ls /
docker exec namenode hdfs dfs -rm -r /test
```

Access the NameNode Web UI in your browser at:
[http://localhost:9870](http://localhost:9870)

### 5. Stop Services
- **Pause/Stop containers without deleting state:**
  ```bash
  docker compose stop
  ```
- **Stop and remove containers while preserving volumes and data:**
  ```bash
  docker compose down
  ```
- **Stop containers and completely wipe all volumes and stored data:**
  ```bash
  docker compose down -v
  ```

---

## 7. Troubleshooting Tips

1. **Port Conflicts**:
   - Verify ports `9092`, `9870`, `9000`, or `9864` are not occupied by local instances or other containers.
   - Run `netstat -ano | findstr 9092` in Windows Command Prompt/PowerShell to find conflicting process IDs.

2. **NameNode Safe Mode**:
   - Immediately upon starting, NameNode enters safe mode until DataNode reports its block inventory. Safe mode automatically turns off within 10–30 seconds.
   - Check status: `docker exec namenode hdfs dfsadmin -safemode get`
   - If needed during development, manually leave safe mode: `docker exec namenode hdfs dfsadmin -safemode leave`

3. **Docker Desktop Memory Allocation**:
   - Running Kafka and Hadoop concurrently requires at least 4 GB of RAM allocated to WSL2/Docker Desktop.
   - You can configure this in `%USERPROFILE%\.wslconfig`:
     ```ini
     [wsl2]
     memory=6GB
     ```

4. **WebHDFS Redirection for Python Clients on Windows**:
   - When writing files via WebHDFS (`http://localhost:9870/webhdfs/v1`), the NameNode responds with an HTTP 307 redirect pointing to the DataNode.
   - The DataNode HTTP port `9864` is published to the Windows host to support host-based WebHDFS clients.

---

## 8. Planned Project Phases

- **Phase 1: Project Structuring & Environment Setup** *(Completed)*
  - Establish directory structure, dependencies (`requirements.txt`), Docker Compose (`docker-compose.yml`), and documentation.
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
  - Final integration testing, reproducible deployment report, and presentation materials.

---

## 9. Directory Layout

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
├── docker/                   # Docker environment configuration (e.g. hadoop.env)
│
├── docs/                     # Project documentation, architecture diagrams, and reports
│
├── docker-compose.yml        # Multi-container Kafka (KRaft) and Hadoop HDFS cluster
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and roadmap
└── .gitignore                # Version control ignore rules
```
