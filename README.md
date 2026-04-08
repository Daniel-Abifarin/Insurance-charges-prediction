# Insurance-charges-prediction
Predicting medical insurance charges using Linear Regression, Random Forest and XGBoost
# Insurance Charges Prediction

## Overview
This project predicts medical insurance charges for individuals based on 
demographic and lifestyle features. Three machine learning models were 
built, compared, and evaluated to identify the best performing approach.

## Dataset
- **Source:** Medical Cost Personal Dataset
- **Size:** 1,338 records, 7 features
- **Target variable:** charges (USD)
- **Features:** age, sex, bmi, children, smoker, region

## Project Structure
├── insurance_charges_prediction.ipynb  # Main notebook
├── insurance.csv                        # Dataset
└── README.md                           # Project documentation

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Methodology

### 1. Exploratory Data Analysis
- Checked for null values and data types
- Visualised distributions of all features
- Computed correlation between features and target variable
- Identified charges as right-skewed — log transformation applied

### 2. Data Preprocessing
- Encoded binary columns (sex, smoker) using label encoding
- Applied one-hot encoding to region column (drop_first=True 
  to avoid dummy variable trap)
- Applied np.log1p() to charges to normalise the distribution

### 3. Key EDA Findings
- Smoking status is the strongest predictor of charges (r = 0.79)
- Age (r = 0.30) and BMI (r = 0.20) are secondary predictors
- Sex and region showed minimal correlation with charges (< 0.08)
- The gap between smokers and non-smokers in charges is non-linear,
  which explains why linear regression underperforms

### 4. Models Built
Three models were trained and evaluated:
- Linear Regression (baseline)
- Random Forest
- XGBoost with GridSearchCV hyperparameter tuning

### 5. Hyperparameter Tuning
GridSearchCV with 5-fold cross validation was used to tune XGBoost 
across the following parameters:
- n_estimators: [100, 200, 300]
- learning_rate: [0.05, 0.1, 0.2]
- max_depth: [3, 5, 7]
- subsample: [0.8, 1.0]

Best parameters found: learning_rate=0.05, max_depth=3, 
n_estimators=100, subsample=0.8

## Results

### Test Set Performance
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.6067 | $3,888 | $7,814 |
| Random Forest | 0.8776 | $2,088 | $4,359 |
| XGBoost (Default) | 0.8563 | $2,220 | $4,723 |
| XGBoost (Tuned) | 0.8792 | $1,939 | $4,331 |

### 5-Fold Cross Validation
| Model | CV Mean R² | Std Dev |
|---|---|---|
| Linear Regression | 0.7638 | 0.0502 |
| Random Forest | 0.8061 | 0.0621 |
| XGBoost Tuned | 0.8366 | 0.0571 |

## Key Findings
- Tuned XGBoost is the best performing model with R² of 0.88 and 
  average prediction error of $1,939
- Default XGBoost underperformed Random Forest, but hyperparameter 
  tuning closed the gap and pushed it to the top
- Feature importance analysis confirmed that smoker, age, and bmi 
  are the three most important predictors in both Random Forest and 
  XGBoost — consistent with EDA correlation findings
- XGBoost assigns 0.65 importance to smoker alone, showing it relies 
  on this feature more aggressively than Random Forest (0.44)
- All region columns and sex contributed negligible importance in 
  both models

## What I Would Do Next
- Deploy the best model as a REST API using FastAPI
- Build a Streamlit web app for interactive predictions
- Explore SHAP values for deeper model explainability
- Collect more data to improve performance on edge cases
- Try stacking Random Forest and XGBoost as an ensemble

## Author
Daniel Abifarin
Electrical Engineering Student | University of Lagos
Aspiring MLOps Engineer
