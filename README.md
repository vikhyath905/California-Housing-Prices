# 🏡 California Housing Price Prediction (ML Regression Project)

This project focuses on analyzing and predicting housing prices in California using machine learning techniques. The dataset is derived from the 1990 U.S. Census and includes various socio-economic and geographical features. The goal is to predict the **median house value** for each block group using regression algorithms and feature selection techniques.

---

## 📁 Dataset Overview

The dataset includes **20,640 samples**, each representing a block group in California, with the following features:

| Feature       | Description                                        |
|---------------|----------------------------------------------------|
| MedInc        | Median income in the block group                   |
| HouseAge      | Median house age in the block group                |
| AveRooms      | Average number of rooms per household              |
| AveBedrms     | Average number of bedrooms per household           |
| Population    | Total population in the block group                |
| AveOccup      | Average number of household members                |
| Latitude      | Block group latitude                               |
| Longitude     | Block group longitude                              |

**Target variable**: `MedHouseVal` — Median house value (in $100,000s)

---

## 📊 Project Workflow

### 1. 📥 Data Acquisition & Exploration
- Dataset loaded using `fetch_california_housing` from Scikit-learn.
- Exploratory data analysis (EDA) performed to:
  - Visualize feature distributions.
  - Compute and plot correlation matrix (heatmap).
  - Check for outliers and relationships with the target.

### 2. 🛠️ Data Preparation
- No missing values in the dataset.
- Feature engineering and transformation (e.g., interaction terms).
- Standardization using `StandardScaler`.
- Data split into **training and test sets** using `train_test_split`.

### 3. 🤖 Modeling
#### ✅ Linear Regression
- Implemented a basic linear regression model.
- Evaluated using both test data and cross-validation.

#### ✅ Random Forest Regressor
- Ensemble-based Random Forest model for better performance.
- Applied `RandomizedSearchCV` for basic hyperparameter tuning.

---

## 📈 Model Performance

| Model            | Mean Squared Error | R-squared | CV R-squared |
|------------------|--------------------|-----------|--------------|
| Linear Regression | 0.5559             | 0.5758    | 0.6115       |
| Random Forest     | 0.2556             | 0.8050    | 0.8042       |

> ✅ **Random Forest outperformed linear regression** significantly in terms of both accuracy and cross-validation score.

---

## 🔍 Feature Importance

### Linear Regression Coefficients
| Feature     | Coefficient |
|-------------|-------------|
| MedInc      | 0.854       |
| AveBedrms   | 0.339       |
| HouseAge    | 0.123       |
| Population  | -0.002      |
| AveOccup    | -0.041      |
| AveRooms    | -0.294      |
| Longitude   | -0.870      |
| Latitude    | -0.897      |

### Random Forest Feature Importance
| Feature     | Importance  |
|-------------|-------------|
| MedInc      | 0.525       |
| AveOccup    | 0.138       |
| Latitude    | 0.089       |
| Longitude   | 0.089       |
| HouseAge    | 0.055       |
| AveRooms    | 0.044       |
| Population  | 0.031       |
| AveBedrms   | 0.030       |

---

## 🔬 Feature Selection (RFE)

Recursive Feature Elimination (RFE) was applied to Linear Regression to find the most influential predictors:

**Selected Features:**
- MedInc
- AveRooms
- AveBedrms
- Latitude
- Longitude

> 🎯 These features contributed the most to model accuracy while simplifying the model.

---

## 📦 Tech Stack

- **Language**: Python
- **Libraries**: Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **Models**: Linear Regression, Random Forest Regressor
- **Feature Selection**: Recursive Feature Elimination (RFE)
- **Validation**: Train/Test split, K-Fold Cross-Validation
