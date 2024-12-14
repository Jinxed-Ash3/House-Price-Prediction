# House Price Prediction

## Overview
This project implements a comprehensive machine learning pipeline to predict house prices using various regression models. The pipeline includes data preprocessing, feature engineering, train-test splitting, modeling, and hyperparameter tuning. The models evaluated include:

1. **Linear Regression**
2. **Lasso Regression**
3. **Random Forest**
4. **XGBoost**
5. **Support Vector Regression (SVR)**

Performance is evaluated using metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared.

---
## Table of Contents
1. [Libraries and Setup](#libraries-and-setup)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Cleaning and Transformation](#data-cleaning-and-transformation)
4. [Feature Engineering](#feature-engineering)
5. [Train-Test Split](#train-test-split)
6. [Modeling and Evaluation](#modeling-and-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Results](#results)

---
## 1. Libraries and Setup
The following libraries are used for data manipulation, visualization, and machine learning:
- **dplyr**, **tidyverse**: Data manipulation
- **lubridate**: Date handling
- **ggplot2**, **corrplot**: Data visualization
- **glmnet**: Lasso regression
- **randomForest**: Random Forest model
- **xgboost**: XGBoost model
- **e1071**: Support Vector Regression
- **caret**: Model tuning

To install these libraries:
```r
install.packages(c("dplyr", "tidyverse", "lubridate", "ggplot2", "corrplot", "glmnet", "randomForest", "xgboost", "e1071", "caret"))
```

---
## 2. Data Loading and Exploration
- The dataset `urban_housing_dataset.csv` is loaded and explored.
- **Key Steps:**
  - Initial inspection using `head()` and `summary()`.
  - Visualizations:
    - Distribution of the target variable (`price`) using histograms.
    - Histograms for numerical features.
    - Correlation heatmap for numerical features.

---
## 3. Data Cleaning and Transformation
- **Handling Missing Values:**
  - Missing values are imputed using the median of each column.
- **Data Scaling and Encoding:**
  - Numerical features are scaled to the range [0,1].
  - Categorical features are encoded as numeric factors.
- A final cleaned dataset is prepared for modeling.

---
## 4. Feature Engineering
- Key features for prediction are selected, including:
  - `square_footage`, `bedrooms`, `bathrooms`, `year_built`, and `crime_rate`.
- Polynomial and interaction features are added:
  - `crime_rate_squared`, `year_built_squared`, and an interaction term (`square_footage * crime_rate`).

---
## 5. Train-Test Split
- The data is split into training (70%) and testing (30%) sets.
- **Features and Target:**
  - Features: Selected and engineered variables.
  - Target: `price`.

---
## 6. Modeling and Evaluation
The following models are implemented and evaluated:

### Linear Regression
- **Baseline model** using all features.
- Stepwise regression is applied for feature selection.

### Lasso Regression
- Lasso regularization is applied using **glmnet**.
- Optimal lambda is determined via cross-validation.

### Random Forest
- Random Forest is trained with 500 trees.
- Tuning is performed using cross-validation to optimize `mtry`.

### XGBoost
- XGBoost model is trained with varying hyperparameters:
  - `max_depth`: {3, 5, 7}
  - `eta`: {0.01, 0.1, 0.3}
  - `nrounds`: {50, 100, 200}.

### Support Vector Regression (SVR)
- SVR with **radial kernel** is applied.
- Hyperparameter tuning grid:
  - `C`: {0.1, 1, 10}
  - `epsilon`: {0.01, 0.1, 1}
  - `gamma`: {0.001, 0.01, 0.1}.

---
## 7. Hyperparameter Tuning
Hyperparameter tuning is performed for:
1. **Random Forest** using the `caret` package with cross-validation.
2. **XGBoost** by grid search.
3. **SVR** with a tuning grid for cost (`C`), epsilon, and gamma.

---
## 8. Results
The performance of all models is evaluated using the following metrics:
- **RMSE**: Measures prediction error.
- **MAE**: Measures average absolute error.
- **R-squared**: Indicates model fit.

### Model Comparison:
| Model               | RMSE    | MAE     | R-squared |
|---------------------|---------|---------|-----------|
| Linear Regression   | 0.0938  | 0.0755  | 52.4%     |
| Random Forest       | 0.0982  | 0.0792  | 47.7%     |
| XGBoost             | 0.1030  | 0.0831  | 42.5%     |
| Support Vector Reg. | 0.0970  | 0.0774  | 48.9%     |

### Key Insights:
- **Linear Regression** performed the best, suggesting a strong linear relationship between predictor variables and house prices.
- **SVR** performed competitively, demonstrating its ability to capture non-linear relationships.
- **Random Forest** and **XGBoost** underperformed slightly, indicating the need for further tuning or additional features.

---
## 9. Conclusion
This project demonstrates the implementation of multiple regression models for house price prediction. Key highlights include:
- Effective use of feature engineering and polynomial terms.
- Model evaluation and hyperparameter tuning.
- Insights into the linearity of the dataset and model performance.

Future work may include:
- Incorporating additional features (e.g., location, neighborhood quality).
- Exploring ensemble methods or hybrid models for improved performance.

---
## 10. Instructions to Run
1. Install required libraries.
2. Load the dataset `urban_housing_dataset.csv`.
3. Run the R script step by step in an IDE like RStudio.
4. Review the evaluation metrics printed in the console.

---
## Contact
For any questions or clarifications, please reach out via email or open an issue in the repository.

