# ✈️ Flight Delay Prediction using PySpark & Deep Learning

A comprehensive data science project that performs **Exploratory Data Analysis (EDA)** and **flight delay prediction** on a real-world airline dataset. The project compares five machine learning models — including traditional ML models via PySpark and deep learning models (CNN, LSTM) via TensorFlow/Keras.

---

## 📌 Overview

Flight delays are a major pain point in the airline industry. This project aims to:
- Understand patterns and causes of flight delays through EDA
- Build and compare multiple predictive models for **arrival delay (ArrDelay)**
- Evaluate models using regression and classification metrics

---

## 📊 Models Implemented

| Model | Framework |
|---|---|
| Linear Regression | PySpark MLlib |
| Random Forest Regressor | PySpark MLlib |
| Gradient-Boosted Trees (GBT) | PySpark MLlib |
| CNN (1D Convolutional Neural Network) | TensorFlow / Keras |
| LSTM (Long Short-Term Memory) | TensorFlow / Keras |

---

## 📁 Dataset

The project uses `Flight_delay.csv` with the following key features:

| Column | Description |
|---|---|
| `DayOfWeek` | Day of the week (1=Mon ... 7=Sun) |
| `DepTime` / `ArrTime` | Departure / Arrival time |
| `ArrDelay` / `DepDelay` | Arrival / Departure delay in minutes |
| `Airline` / `UniqueCarrier` | Airline name and code |
| `Origin` / `Dest` | Origin and destination airport codes |
| `Distance` | Flight distance in miles |
| `CarrierDelay` | Delay caused by the carrier |
| `WeatherDelay` | Delay caused by weather |
| `NASDelay` | Delay caused by National Airspace System |
| `SecurityDelay` | Delay caused by security |
| `LateAircraftDelay` | Delay caused by a late arriving aircraft |

> **Note:** The dataset file `Flight_delay.csv` is not included in this repo due to size. You can source a similar dataset from [Kaggle - Airline Delay and Cancellation Data](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018).

---

## 📈 EDA Highlights

- Missing value analysis with bar chart visualization
- Distribution of arrival delays (histogram + KDE)
- Box plots for arrival vs departure delays
- Flight count per airline
- Correlation heatmap of numeric features
- Pair plots for key numeric features
- Delay percentage breakdown by **day of week**, **year**, and **delay reason**

---

## 🤖 ML Pipeline (PySpark)

1. Load and parse the CSV into a Spark DataFrame
2. Encode categorical columns using `StringIndexer`
3. Assemble feature vector with `VectorAssembler`
4. Train/test split: **70% / 30%**
5. Train all three PySpark models via `Pipeline`
6. Evaluate using RMSE, R², MAE, AUC-ROC, Precision, Recall, F1

---

## 🧠 Deep Learning Models (TensorFlow/Keras)

- **CNN:** 1D Convolutional layer → Flatten → Dense layers
- **LSTM:** LSTM layer → Dense output layer
- Both trained for **10 epochs** with batch size **32**
- Input: 10 numeric features reshaped for sequential models

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| MAE | Mean Absolute Error |
| AUC-ROC | Area Under ROC Curve (delay > 15 min threshold) |
| Precision / Recall | Binary classification metrics |
| F1 Score | Harmonic mean of Precision & Recall |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **PySpark** — distributed ML pipelines
- **TensorFlow / Keras** — CNN and LSTM models
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualization
- **scikit-learn** — additional metrics (AUC-ROC, F1)

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install pyspark tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 2. Add Dataset

Place `Flight_delay.csv` in the project root directory.

### 3. Run the Notebook

```bash
jupyter notebook fiverr.ipynb
```

Or run as a Python script:

```bash
python fiverr.py
```

---

## 📂 Project Structure

```
flight-delay-prediction/
├── fiverr.ipynb          # Main Jupyter Notebook (EDA + ML + DL)
├── Flight_delay.csv      # Dataset (not included — see Dataset section)
└── README.md
```

---

## 👤 Author

**sohaib0075**  
[GitHub Profile](https://github.com/sohaib0075)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
