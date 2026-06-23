# 🌧️ AP Rainfall Early Warning System

An AI-powered rainfall monitoring and early warning platform for Andhra Pradesh that predicts rainfall risk levels using Machine Learning and visualizes vulnerable locations through an interactive dashboard.

![Dashboard](screenshots/dashboard.png)

---

## 📌 Project Overview

The AP Rainfall Early Warning System is a data-driven disaster preparedness platform developed to monitor and predict rainfall risk across Andhra Pradesh.

The system leverages historical rainfall records, geospatial infrastructure data, and a Random Forest Machine Learning model to generate risk alerts at the mandal level. Results are presented through a modern Streamlit dashboard featuring interactive maps, rainfall analytics, canal monitoring, and embankment risk assessment.

The objective is to assist authorities, irrigation departments, and disaster management teams in identifying vulnerable locations before severe rainfall events occur.

---

## 🚀 Features

### 🌦 Rainfall Prediction
- Predicts rainfall intensity using Machine Learning.
- Generates seasonal rainfall forecasts.
- Provides district-wise and mandal-wise predictions.

### ⚠️ Risk Classification
Automatically categorizes locations into:

| Risk Level | Rainfall Range |
|------------|---------------|
| 🟢 Green | < 15.6 mm |
| 🟡 Yellow | 15.6 – 64.4 mm |
| 🟠 Orange | 64.5 – 115.5 mm |
| 🔴 Red | ≥ 115.6 mm |

### 🗺 Interactive Risk Map
- Andhra Pradesh-wide visualization.
- Color-coded risk markers.
- Geographic risk assessment.

### 📊 District & Mandal Analytics
- Detailed rainfall statistics.
- Mandal-wise risk information.
- Dynamic filtering options.

### 🌊 Canal Monitoring
- Canal risk assessment.
- Infrastructure vulnerability tracking.
- District-wise canal monitoring.

### 🏞 Embankment Monitoring
- Flood-sensitive embankment analysis.
- Riverbank vulnerability assessment.
- Early warning support.

### ⚡ Real-Time Alert Generation
- Instant rainfall risk generation.
- Date-based forecasting.
- Automated risk categorization.

---

# 🏗️ System Architecture

```text
Historical Rainfall Data
           │
           ▼
Data Cleaning & Preprocessing
           │
           ▼
Feature Engineering
           │
           ▼
Random Forest Model Training
           │
           ▼
Rainfall Prediction
           │
           ▼
Risk Classification
           │
           ▼
Canal & Embankment Assessment
           │
           ▼
Interactive Streamlit Dashboard
```

---

# 🤖 Machine Learning Model

### Model Used
- Random Forest Machine Learning Model

### Why Random Forest?
- Handles non-linear relationships effectively
- Robust against noisy environmental data
- High prediction stability
- Suitable for rainfall forecasting problems

### Prediction Outputs
- Rainfall Forecast
- Risk Level Classification
- Canal Risk Assessment
- Embankment Risk Assessment

---

# 📂 Dataset Information

### Source
Historical Andhra Pradesh Rainfall Dataset

### Coverage
- Andhra Pradesh State
- 28 Districts
- 215+ Mandals

### Data Used
- Historical Rainfall Records
- Canal Infrastructure Data
- Embankment Infrastructure Data
- Geographic Location Data

### Training Period
- 3 Years Historical Rainfall Data

---

# 🛠️ Technology Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn
- Random Forest

### Dashboard
- Streamlit

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly
- Folium
- Leaflet

### Storage
- CSV
- Parquet
- Pickle Models

---

# 📂 Project Structure

```text
AP-RAINFALL-EARLY-WARNING-SYSTEM/
│
├── Canals/
│   ├── Canals.shp
│   ├── Canals.dbf
│   ├── Canals.shx
│   └── ...
│
├── Embankments/
│   ├── Embankments.shp
│   ├── Embankments.dbf
│   ├── Embankments.shx
│   └── ...
│
├── IMD_AP_Historical Rain Fall/
│   ├── AP_Village_Daily_Rainfall_21_r.csv
│   ├── AP_Village_Daily_Rainfall_22_r.csv
│   ├── AP_Village_Daily_Rainfall_23_r.csv
│   ├── AP_Village_Daily_Rainfall_24_r.csv
│   ├── AP_Village_Daily_Rainfall_25_r.csv
│   └── *.parquet
│
├── models/
│   ├── rf_model.pkl
│   ├── master_dataset.parquet
│   ├── village_lookup.parquet
│   ├── feature_cols.pkl
│   └── label_encoder.pkl
│
├── screenshots/
│   ├── dashboard.png
│   ├── risk_map.png
│   ├── rainfall_analysis.png
│   ├── canal_monitoring.png
│   └── embankment_monitoring.png
│
├── predictor.py
├── streamlit_app.py
├── convert_to_parquet.py
├── requirements.txt
└── README.md
```

---

# 📸 Application Screenshots

## Dashboard Overview

Displays overall rainfall risk distribution, alert levels, and prediction controls.

![Dashboard](screenshots/dashboard.png)

---

## Interactive Village Risk Map

Visual representation of rainfall risk levels across Andhra Pradesh.

![Risk Map](screenshots/risk_map.png)

---

## District & Mandal Rainfall Analysis

Provides detailed rainfall prediction results and risk classifications.

![Rainfall Analysis](screenshots/rainfall_analysis.png)

---

## Canal Risk Monitoring

District-wise canal vulnerability analysis based on rainfall forecasts.

![Canal Monitoring](screenshots/canal_monitoring.png)

---

## Embankment Risk Monitoring

Monitors embankments and flood-prone regions for potential threats.

![Embankment Monitoring](screenshots/embankment_monitoring.png)

---

# 📊 Dashboard Modules

## 1. Risk Overview

Displays:

- Extreme Risk Count
- High Risk Count
- Moderate Risk Count
- No Risk Count
- Overall Alert Status
- Total Mandals Assessed

---

## 2. Village Risk Map

Interactive map showing:

- Risk locations
- Color-coded alerts
- Geographic distribution

---

## 3. Rainfall Data Analysis

Displays:

- District
- Mandal
- Rainfall Prediction
- Risk Category

---

## 4. Canal Risk Assessment

Provides:

- Canal Name
- District
- Canal Type
- Risk Status

---

## 5. Embankment Risk Assessment

Provides:

- Embankment Name
- District
- River Name
- Risk Status

---

# 📈 Model Performance

### Coverage

- 28 Districts
- 215+ Mandals
- 1200+ Canals
- 40+ Embankments

### Prediction Outputs

✔ Rainfall Forecasting

✔ Risk Classification

✔ Infrastructure Risk Monitoring

✔ Early Warning Alerts

---

# 🌍 Use Cases

- Disaster Management Authorities
- Flood Monitoring Teams
- Irrigation Departments
- Canal Management Authorities
- District Administration
- Environmental Monitoring Agencies
- Research and Planning Organizations

---

# 🔮 Future Enhancements

- Live Weather API Integration
- IMD Real-Time Data Integration
- Satellite Data Support
- SMS Alert System
- Mobile Application
- Deep Learning Models
- Automated Emergency Notifications
- Flood Forecasting Module

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/MahaLakshmiBusi22/AP-RAINFALL-EARLY-WARNING-SYSTEM.git
```

## Navigate to Project

```bash
cd AP-RAINFALL-EARLY-WARNING-SYSTEM
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run streamlit_app.py
```

Application will start at:

```text
http://localhost:8501
```

---

# 💡 Project Highlights

✅ Machine Learning Powered Predictions

✅ Interactive GIS-Based Risk Mapping

✅ Random Forest Forecasting Model

✅ District & Mandal Level Analysis

✅ Canal Monitoring System

✅ Embankment Risk Monitoring

✅ Early Warning Dashboard

✅ Andhra Pradesh Wide Coverage

✅ Streamlit-Based Modern UI

✅ Data-Driven Disaster Preparedness

---

# 👨‍💻 Developer

**Mahalakshmi Busi**

GitHub:
https://github.com/MahaLakshmiBusi22

---

# 📄 License

This project is intended for educational, research, and disaster management purposes.

---

⭐ If you found this project useful, please consider giving it a Star on GitHub.
