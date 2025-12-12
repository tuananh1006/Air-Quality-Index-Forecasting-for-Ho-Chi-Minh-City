# Air-Quality-Index-Forecasting-for-Ho-Chi-Minh-City

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Introduction
This repository contains the source code and dataset for the research project **"Air Quality Index Forecasting for Ho Chi Minh City"**. [cite_start]The project addresses the critical need for real-time air quality monitoring in one of Southeast Asia's most densely populated cities[cite: 19, 30].

The objective is to develop high-accuracy **hourly AQI forecasting models** using advanced Deep Learning architectures (including Hybrid and Decomposition-based models) to support early warning systems for urban pollution.

## 📊 Dataset Overview
We constructed a massive-scale dataset specifically for this study, offering significantly higher resolution than prior works.

* **Data Source:** Weatherbit API & OpenStreetMap.
* **Time Period:** January 2023 – September 2025.
* **Temporal Resolution:** Hourly.
* **Scale:** **941,433 records** across **40 monitoring stations**.
* **Features:**
    * *Pollutants:* AQI, CO, NO2, O3, SO2, PM10, PM2.5.
    * *Meteorological:* Temperature, Humidity, Pressure, Wind Speed, Clouds, Precipitation, UV Index.

## 🛠 Methodology

### 1. Data Pipeline
The data acquisition and preparation process follows a rigorous automated pipeline:
1.  **Crawling & Geocoding:** Scraped administrative units from *thuvienphapluat.vn* and mapped them to geographic coordinates using the OpenStreetMap API.
2.  **Retrieval:** Extracted historical weather and air quality data via Weatherbit API.
3.  **Preprocessing:**
    * **Outlier Detection:** Applied **Isolation Forest** to robustly handle anomalies.
    * **Imputation:** Used cubic spline interpolation for missing values.
    * **Normalization:** Scaled features to ensure model stability.

### 2. Models Implemented
We implemented and compared **9 distinct architectures** to evaluate their capability in capturing nonlinear and spatiotemporal dependencies:

* **Sequential Baselines:** GRU, LSTM, BiLSTM.
* **Hybrid Deep Learning:** **BiLSTM-CNN** (Combines temporal learning with feature extraction).
* **Graph-Based:** STGCN, STGCN-Adaptive.
* **Decomposition Hybrids:** VMD-CNN-LSTM, EEMD-CNN-LSTM.
* **Attention-Based:** Transformer.

## 📈 Experimental Results

The models were evaluated using 7 standard regression metrics. Below is the detailed performance comparison:

| Model | MAE | MSE | RMSE | MSLE | RMSLE | MAPE | $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BiLSTM-CNN** | **5.9362** | **193.0403** | **13.8939** | **0.0433** | **0.2082** | **12.17** | 0.7961 |
| GRU | 6.2745 | 198.6965 | 14.0960 | 0.0459 | 0.2141 | 15.16 | 0.7901 |
| LSTM | 6.7147 | 332.2190 | 18.2269 | 0.1108 | 0.3329 | 12.82 | 0.6491 |
| BiLSTM | 7.0079 | 334.1439 | 18.2796 | 0.0729 | 0.2699 | 13.83 | 0.6471 |
| VMD-CNN-LSTM | 9.0841 | 157.4126 | 12.5464 | 0.1363 | 0.3692 | 37.23 | **0.8337** |
| Transformer | 12.0680 | 286.4227 | 16.9240 | 0.1265 | 0.3557 | 36.15 | 0.6975 |
| STGCN | 10.7040 | 427.8774 | 20.6852 | 0.1545 | 0.3930 | 29.09 | 0.5344 |
| STGCN-Adaptive | 10.3761 | 423.6522 | 20.5828 | 0.1519 | 0.3897 | 27.01 | 0.5390 |
| EEMD-CNN-LSTM | 13.0486 | 254.6857 | 15.9589 | 0.1964 | 0.4432 | 48.78 | 0.7310 |

*(Bold indicates the best performance in that metric category)*

### 🔍 Key Findings
Based on the experimental analysis, we observed the following interesting patterns:

1.  **BiLSTM-CNN Dominance:** This hybrid model outperformed other methods by ranking **1st in 6/7 metrics** (except R-squared).
2.  **Effectiveness of CNN:** Using CNN layers to reduce parameters from the LSTM architecture appears to be the key factor for the success of BiLSTM-CNN compared to the standard BiLSTM.
3.  **LSTM vs. BiLSTM:** We did **not** recognize any significant performance increase when upgrading from LSTM to BiLSTM in this specific dataset.
4.  **Signal Decomposition:** While BiLSTM-CNN was the most balanced, the **VMD-CNN-LSTM** model achieved the highest $R^2$ score (0.8337), suggesting that signal decomposition is highly effective for explaining variance.