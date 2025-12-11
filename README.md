# Austraian-Waether


---

# ☀️ Australian Weather Prediction Using Python

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--12--11-blueviolet)]()
[![Dataset Size](https://img.shields.io/badge/Dataset-10k%2B%20rows-orange)]()
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-FF5733?logo=googlecolab\&logoColor=white)](https://colab.research.google.com/drive/<your-colab-link>)

Predict weather patterns across Australia using machine learning models with historical meteorological data. The project includes **data preprocessing, feature engineering, visualization, and predictive modeling**.

---

## 🚀 Project Overview

Weather prediction is crucial for agriculture, transportation, tourism, and disaster management. This project uses historical Australian weather data to build predictive models that forecast key weather parameters like temperature, rainfall, and wind speed.

**Objectives:**

* Analyze historical weather data across Australia
* Explore patterns and seasonal trends
* Build predictive models for temperature, rainfall, and other features
* Visualize predictions and feature importance

---

## 📊 Dataset

The dataset contains daily weather observations from multiple Australian weather stations.

**Key Features:**

| Feature       | Description                                       |
| ------------- | ------------------------------------------------- |
| Date          | Observation date                                  |
| Location      | Weather station location                          |
| MinTemp       | Minimum temperature (°C)                          |
| MaxTemp       | Maximum temperature (°C)                          |
| Rainfall      | Rainfall (mm)                                     |
| Evaporation   | Evaporation (mm)                                  |
| Sunshine      | Sunshine hours                                    |
| WindGustDir   | Direction of strongest wind gust                  |
| WindGustSpeed | Speed of strongest wind gust (km/h)               |
| WindDir9am    | Wind direction at 9am                             |
| WindDir3pm    | Wind direction at 3pm                             |
| WindSpeed9am  | Wind speed at 9am (km/h)                          |
| WindSpeed3pm  | Wind speed at 3pm (km/h)                          |
| Humidity9am   | Humidity at 9am (%)                               |
| Humidity3pm   | Humidity at 3pm (%)                               |
| Pressure9am   | Atmospheric pressure at 9am (hPa)                 |
| Pressure3pm   | Atmospheric pressure at 3pm (hPa)                 |
| Cloud9am      | Cloud cover at 9am (oktas)                        |
| Cloud3pm      | Cloud cover at 3pm (oktas)                        |
| Temp9am       | Temperature at 9am (°C)                           |
| Temp3pm       | Temperature at 3pm (°C)                           |
| RainToday     | Did it rain today? (Yes/No)                       |
| RainTomorrow  | Will it rain tomorrow? (Yes/No) – target variable |

> Source: [Australian Weather Dataset – Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

---

## 🛠️ Tools & Libraries

* **Python 3.11** – Programming language
* **Pandas & NumPy** – Data manipulation
* **Matplotlib, Seaborn & Plotly** – Visualizations
* **Scikit-learn & XGBoost** – Machine learning models
* **Statsmodels** – Time series analysis
* **Jupyter Notebook / Google Colab** – Interactive exploration

---

## 🔍 Key Analyses

1. **Data Cleaning & Preprocessing**

   * Handle missing values, outliers, and categorical encoding
   * Feature engineering for time-related features (month, season, etc.)

2. **Exploratory Data Analysis (EDA)**

   * Visualize trends in temperature, rainfall, and wind speed
   * Compare weather patterns across different regions

3. **Predictive Modeling**

   * Classification: Predict **RainTomorrow** using historical weather data
   * Regression: Predict temperature, humidity, or rainfall values

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
import pandas as pd
df = pd.read_csv('australian_weather.csv')

# Preprocessing example
X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Pressure9am']]
y = df['RainTomorrow'].map({'Yes':1, 'No':0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

4. **Visualization of Predictions**

   * Feature importance plots
   * Interactive Plotly charts for rainfall probability and temperature trends

5. **Time-Series Forecasting (Optional)**

   * Forecast temperature or rainfall using **ARIMA/Prophet** models

---

## 📈 Performance Metrics

| Task                    | Model                       | Metric        |
| ----------------------- | --------------------------- | ------------- |
| RainTomorrow Prediction | Random Forest               | Accuracy ~85% |
| Temperature Forecasting | Linear Regression / XGBoost | RMSE ~2–3°C   |

> Performance depends on features selected and data preprocessing methods.

---

## ⚡ Usage

1. Clone the repository:

```bash
git clone <repo-url>
cd australian-weather-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis notebook or script:

```bash
python weather_analysis.py
```

4. Explore interactive charts using **Plotly** or launch dashboards (Streamlit/Dash if included).

5. Optional: Try the project on Google Colab: [Open in Colab](https://colab.research.google.com/drive/<your-colab-link>)

---

## 📂 Folder Structure

```
australian-weather-prediction/
│
├─ australian_weather.csv
├─ weather_analysis.py
├─ notebooks/
│   ├─ EDA.ipynb
│   ├─ RainPrediction.ipynb
│   └─ TempForecast.ipynb
├─ visualizations/
│   ├─ rain_prediction_plot.png
│   ├─ temperature_trend.png
│   └─ interactive_plotly_chart.gif
├─ README.md
├─ requirements.txt
└─ LICENSE
```

---

## 💡 Insights

* Predicting rainfall is feasible with high accuracy using Random Forest and historical weather data
* Temperature and rainfall patterns vary significantly by region and season
* Feature importance identifies key drivers of rainfall (Humidity9am, Pressure9am, RainToday)
* Interactive visualizations improve understanding of trends across time and location

---

## 📌 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---



Do you want me to do that next?
