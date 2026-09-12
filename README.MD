# 🚌 TransitAI

## AI-Powered Public Transport Demand Intelligence & Forecasting

## 🌐 Live Application

### 🚀 Try TransitAI

**Live Demo:**  
https://transitai-ckbfrhivnkxqgcz5aqdew9.streamlit.app/

TransitAI is deployed as an interactive Streamlit application where users can select a bus route and explore predicted passenger demand for the next 24 hours.

---

# 📌 Overview

TransitAI is an **AI-powered public transportation demand forecasting and intelligence platform**.

The project uses machine learning to transform historical transportation demand data into short-term passenger-demand forecasts.

Instead of presenting predictions as raw numbers alone, TransitAI converts the forecast into operational intelligence such as:

- Expected passenger demand
- Peak-demand periods
- Lowest-demand periods
- Average demand
- Demand-level classification
- High-demand alerts
- Capacity-planning recommendations
- Hour-by-hour forecast tables

The goal is to demonstrate how a machine-learning forecasting system can move beyond prediction and provide information that can support **public transportation planning and operational decision-making**.

---

# 🎯 Problem Statement

Public transportation systems experience demand fluctuations throughout the day.

Passenger demand can vary significantly depending on:

- Time of day
- Weekday vs. weekend
- Peak vs. non-peak periods
- Route characteristics
- Historical demand patterns

Without reliable short-term demand forecasts, transport operators may face challenges such as:

- Insufficient capacity during peak periods
- Underutilized vehicles during low-demand periods
- Inefficient fleet allocation
- Difficulty identifying upcoming demand surges
- Limited visibility into future passenger demand

TransitAI addresses this problem by building a machine-learning forecasting pipeline capable of generating **24-hour passenger-demand forecasts**.

---

# 💡 Solution

TransitAI follows a complete data-science workflow:

```text
Historical Transport Data
          │
          ▼
    Data Extraction
          │
          ▼
     Data Cleaning
          │
          ▼
   Feature Engineering
          │
          ▼
 Time-Series Data Splitting
          │
          ▼
 XGBoost Model Training
          │
          ▼
    Model Evaluation
          │
          ▼
  24-Hour Demand Forecast
          │
          ▼
 Demand-Level Classification
          │
          ▼
Operational Intelligence
          │
          ▼
   Streamlit Dashboard
