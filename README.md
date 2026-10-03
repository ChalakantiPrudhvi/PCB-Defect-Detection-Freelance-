Live Demo :https://pcb-defect-detection-freelance.onrender.com/

# PCB Defect Detection System Using Machine Learning

A machine learning-based web application that predicts whether a Printed Circuit Board (PCB) is **Defective** or **Non-Defective** based on manufacturing and inspection parameters.

The system uses machine learning for classification and provides a simple web interface where users can enter PCB parameters and receive a prediction along with the probability of each class.

---

## Project Overview

PCB quality inspection is an important part of electronics manufacturing. Defects can occur due to factors such as improper soldering, component misalignment, abnormal electrical values, temperature variations, and poor track quality.

This project demonstrates how machine learning can be used to analyze PCB manufacturing parameters and classify a PCB as:

- **Defective**
- **Non-Defective**

The trained model is integrated with a Flask web application that provides predictions through a user-friendly interface.

> **Dataset Note:** The dataset used in this project is a synthetic dataset containing 400 records created for educational and project demonstration purposes. It is not collected from an actual PCB manufacturing facility.

---

## Features

- PCB defect classification using Machine Learning
- Multiple ML algorithms compared
- Data preprocessing using Scikit-learn
- Numerical feature scaling
- Categorical feature encoding
- Class imbalance handling
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Prediction probability
- Flask REST API
- Responsive web interface
- Production deployment using Gunicorn
- Deployable on Render

---

## Technologies Used

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn

### Backend

- Flask
- Gunicorn

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Git
- GitHub
- Render

---

## Dataset

The dataset contains **400 PCB records** with 11 columns.

### Features

| Feature | Description |
|---|---|
| PCB_ID | Unique PCB identifier |
| Voltage_V | PCB voltage |
| Current_A | PCB current |
| Resistance_Ohm | PCB resistance |
| Temperature_C | Operating temperature |
| Solder_Quality | Solder quality score |
| Component_Alignment | Component alignment score |
| Track_Quality | Track quality score |
| Solder_Type | Type of solder used |
| Inspection_Status | Inspection result |
| Defect_Status | Target variable |

### Target Classes

- `Defective`
- `Non-Defective`

### Class Distribution

| Class | Records |
|---|---:|
| Non-Defective | 358 |
| Defective | 42 |

The dataset is therefore imbalanced, so accuracy alone is not sufficient to evaluate the model.

---

## Machine Learning Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Data Preprocessing
   ↓
Train Multiple ML Models
   ↓
Model Evaluation
   ↓
Cross Validation
   ↓
Select Final Model
   ↓
Save Model
   ↓
Flask Web Application
   ↓
PCB Prediction
