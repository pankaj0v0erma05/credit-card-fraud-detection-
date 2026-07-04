# 🛡️ Financial Sentinel: Real-Time Credit Card Fraud Detection Platform

Financial Sentinel is a production-ready, end-to-end Machine Learning system engineered to detect fraudulent credit card transactions in real time. The platform integrates a stratified preprocessing pipeline, a benchmarking suite comparing 6 classifiers, and interactive serving gateways (a FastAPI REST API and a Streamlit analytics dashboard).

---

## 📊 Key Platform Metrics (Tuned XGBoost Champion)

| Metric | Training Set | Testing Set | Generalization Gap | Status |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | 99.66% | **85.53%** | 14.13% | 🟢 PASS |
| **F1-Score** | 90.68% | **78.47%** | 12.21% | 🟢 PASS |
| **Recall (Catch Rate)** | 100.00% | **83.67%** | 16.33% | 🟢 PASS |
| **Precision** | 82.95% | **73.87%** | 9.08% | 🟢 PASS |
| **False Alarm Rate** | - | **0.05%** | - | 🟢 PASS |

---

## 🛠️ Project Directory Layout

```
project1_fraud_detection/
├── config/
│   └── config.yaml                     # Baseline project configurations
├── data/                               # Directory containing raw datasets (ignored in Git)
├── artifacts/                          # Pickled preprocessors, model binaries, and arrays
├── static/                             # Static visual graphs and plots for reports
├── src/
│   ├── data_ingestion.py               # Ingestion streams downloading raw source CSVs
│   ├── preprocessing.py                # Stratified splitting and leak-free scaling
│   ├── eda.py                          # Multi-figure Seaborn visualizations generator
│   ├── train.py                        # Leaderboard benchmarking across 6 classifiers
│   ├── tune.py                         # GridSearchCV overfitting control for XGBoost
│   ├── predict.py                      # Production real-time dictionary inference module
│   ├── evaluate.py                     # Validation testing (Confusion Matrix/Class Reports)
│   ├── generate_pdf.py                 # PDF Case Study Report compiler using ReportLab
│   ├── api.py                          # FastAPI ASGI REST app with lifespan caching
│   └── app.py                          # Streamlit Analytical UI Dashboard
├── backend/
│   └── main.py                         # Microservice redirect linking to src/api.py
├── frontend/
│   └── app.py                          # Streamlit dashboard redirect linking to src/app.py
├── case_study_fraud_detection.pdf      # Complete 3-page styled PDF report
└── README.md                           # Platform documentation (this file)
```

---

## ⚙️ Core Architecture & Data Pipeline

### 1. Zero-Leakage Preprocessing
- **Stratification**: Preserves the extreme minority target ratio (**0.173%** fraud vs. **99.827%** legitimate) across train/test slices.
- **ColumnTransformer**: Implements a modular pipeline using `StandardScaler` only for `Time` and `Amount` fields, passing PCA elements (`V1-V28`) through untouched to prevent feature scaling leakage.
- **Fit/Transform Segmentation**: `.fit_transform()` is run exclusively on the training split, while testing and incoming request payloads use `.transform()`.

### 2. Hyperparameter Optimization & Generalization
- The initial baseline XGBoost model achieved 100% training F1 but suffered from overfitting.
- **GridSearchCV** was run with **Stratified 3-Fold CV** targeting **Average Precision (PR-AUC)**.
- Restricting `max_depth` to `5` and `n_estimators` to `150` closed the Train-Test F1 gap from **15.2%** to **12.2%** (PASS) while increasing test recall to **83.67%**.

### 3. Model Validation (Test Confusion Matrix)
Out of 56,962 test transactions:
- **True Negatives** (Legitimate Approved): **56,835**
- **False Positives** (Legitimate Flagged): **29** (only 0.05% false alarm rate)
- **False Negatives** (Fraudulent Missed): **16**
- **True Positives** (Fraudulent Caught): **82**

---

## 📈 Visual Insights (EDA)
The system generates five static plots located in the `static/` directory:
1. **Target Distribution**: Uses log scaling to inspect the extreme class imbalance.
2. **Transaction Value KDE**: Highlights distribution thresholds of legitimate vs. fraud under $250.
3. **Correlation Vector Heatmap**: Isolates features V17, V14, and V12 as the strongest negative correlates.
4. **Diurnal Time Density (Hidden Story 1)**: Shows flat overnight fraud levels (midnight to 5 AM), indicating automated scripts or offshore attacks.
5. **PCA Space separability (Hidden Story 2)**: Visualizes a clear clustering boundary between fraudulent and legitimate points.

---

## 🚀 Installation & Getting Started

### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/pankaj0v0erma05/credit-card-fraud-detection-.git
cd credit-card-fraud-detection-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*(Note: If creating requirements.txt from scratch, install `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `catboost`, `streamlit`, `fastapi`, `uvicorn`, `reportlab`)*

### 2. Execute Data & Modeling Steps
```bash
# Step 1: Download raw ULB transaction CSV
python src/data_ingestion.py

# Step 2: Run stratified split and column scaler transformations
python src/preprocessing.py

# Step 3: Train and benchmark the six baseline models
python src/train.py

# Step 4: Run GridSearch CV to generalize the champion XGBoost model
python src/tune.py

# Step 5: Output test set metrics & confusion matrix
python src/evaluate.py

# Step 6: Generate the styled PDF Case Study Report
python src/generate_pdf.py
```

---

## 🔌 Running the Microservices

### FastAPI REST API (Backend)
The microservice implements an async lifespan context manager to load and cache model binaries in application memory on startup, ensuring quick response times. It validates JSON payloads via Pydantic.
```bash
# Launch server
uvicorn src.api:app --reload --port 8000
```
- **Base Endpoint**: `http://127.0.0.1:8000`
- **Interactive Swagger UI OpenAPI documentation**: `http://127.0.0.1:8000/docs`

### Streamlit Analytical Dashboard (Frontend)
The dashboard provides custom CSS cards, metric widgets, and color indicators to help risk analysts.
```bash
# Launch dashboard
streamlit run src/app.py
```
- **Local URL**: `http://localhost:8501`
- **Features**: 
  - **Single Scan**: Sliders for inputting transaction parameters (real-time prediction results change colors based on risk).
  - **Batch Pipeline**: Drag-and-drop CSV batch transaction scoring with CSV report downloads.
  - **Exploratory Insights**: Interactive review of raw statistical plots and insights.
