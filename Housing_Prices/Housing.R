##Libraries and Data Loading
# Libraries
library(dplyr)
library(tidyverse) 
library(lubridate)
library(ggplot2)
library(ploty)
library(corrplot)
library(kableExtra)
library(fastDummies)
library(glmnet)
library(randomForest)
library(caret)
library(xgboost)
library(e1071)

# Load data
raw.data <- read.csv("C:/Users/Joji/Documents/Housing_Prices/data/urban_housing_dataset.csv")
dim(raw.data)

head(raw.data)

summary(raw.data)

##Data exporation and analysis
# Display first 10 rows in a table format
raw.data[1:10, 1:11] %>%
  kable('pipe', booktabs = T, caption = 'Table 1: Raw Data', format.args = list(big.mark = ',')) %>%
  kable_styling(font_size = 5, latex_options = c('striped', 'hold_position', 'repeat_header'))

# Price distribution histogram
ggplot(raw.data, aes(x = price)) +
  geom_histogram(bins = 30, fill = 'skyblue', color = 'black') +
  theme_minimal() +
  labs(title = "Distribution of Price", x = "Price", y = "Frequency")

# Histograms for all numerical features
raw.data %>%
  select(where(is.numeric)) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = 'skyblue', color = 'black') +
  facet_wrap(~key, scales = 'free', ncol = 3) +
  theme_minimal() +
  labs(title = "Distribution of Numerical Features", x = "Value", y = "Frequency")

# Correlation heatmap
cor_matrix <- cor(raw.data %>% select(where(is.numeric)), use = "complete.obs")
corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.7)

##Data Cleaning and Transformation
# Handle missing values
raw.data <- raw.data %>%
  mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Separate numerical and categorical features
num_data <- raw.data %>% select(where(is.numeric))
cat_data <- raw.data %>% select(where(is.factor) | where(is.character))

# Scale numerical values and encode categorical variables
num_data_scaled <- num_data %>% mutate(across(everything(), ~ (. - min(.)) / (max(.) - min(.))))
cat_data_encoded <- cat_data %>% mutate(across(everything(), ~ as.numeric(as.factor(.))))
final_data <- bind_cols(num_data_scaled, cat_data_encoded)

##Feature Engineering
# Feature selection
target_variable <- "price"
selected_features <- c("square_footage", "bedrooms", "bathrooms", "year_built", "crime_rate")
final_data_selected <- final_data %>% select(all_of(selected_features), target_variable)

# Add polynomial features
final_data_selected <- final_data_selected %>%
  mutate(
    crime_rate_squared = crime_rate^2,
    year_built_squared = year_built^2,
    interaction_term = square_footage * crime_rate
  )

##Train-Test Split
# Split data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(final_data_selected), 0.7 * nrow(final_data_selected))
train_data <- final_data_selected[train_index, ]
test_data <- final_data_selected[-train_index, ]

# Prepare features and target
features <- c("square_footage", "bedrooms", "bathrooms", "crime_rate_squared", "year_built_squared", "interaction_term")
X_train <- as.matrix(train_data[, features])
y_train <- train_data[, target_variable]
X_test <- as.matrix(test_data[, features])
y_test <- test_data[, target_variable]

##Modeling and Evaluation
# Lasso Regression
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda <- lasso_model$lambda.min
cat("Best Lambda for Lasso:", best_lambda, "\n")

#Evaluate the model
evaluate_model <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(list(RMSE = rmse, MAE = mae, R_squared = r_squared))
}

# Predictions and Evaluation for Lasso
y_pred_lasso <- predict(lasso_model, s = "lambda.min", newx = X_test)
lasso_metrics <- evaluate_model(y_test, y_pred_lasso)
print("Lasso Regression Metrics:")
print(lasso_metrics)

# Linear Regression
linear_model <- lm(price ~ ., data = train_data)
linear_pred <- predict(linear_model, test_data)
linear_metrics <- evaluate_model(test_data$price, linear_pred)
print("Linear Regression Metrics:")
print(linear_metrics)

# Random Forest
rf_model <- randomForest(price ~ ., data = train_data, ntree = 500)
rf_pred <- predict(rf_model, test_data)
rf_metrics <- evaluate_model(test_data$price, rf_pred)
print("Random Forest Metrics:")
print(rf_metrics)

# Gradient Boosting (XGBoost)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test)
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)
xgb_pred <- predict(xgb_model, dtest)
xgb_metrics <- evaluate_model(y_test, xgb_pred)
print("XGBoost Metrics:")
print(xgb_metrics)

# Support Vector Regression
svr_model <- svm(price ~ ., data = train_data, kernel = "radial")
svr_pred <- predict(svr_model, test_data)
svr_metrics <- evaluate_model(test_data$price, svr_pred)
print("SVR Metrics:")
print(svr_metrics)

###Hyperparameter Tuning 
##Linear Regression
# Stepwise regression for feature selection
step_model <- step(lm(price ~ ., data = train_data), direction = "both")

# Summary of the best model
summary(step_model)

# Predictions
step_pred <- predict(step_model, newdata = test_data)
step_metrics <- evaluate_model(test_data$price, step_pred)
print("Stepwise Linear Regression Metrics:")
print(step_metrics)

##Random Forest
# Define the tuning grid
rf_grid <- expand.grid(mtry = c(2, 3, 4))  # Varying the number of features

# Train the Random Forest with tuning
rf_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
rf_tuned <- train(
  price ~ ., 
  data = train_data, 
  method = "rf", 
  tuneGrid = rf_grid, 
  trControl = rf_control, 
  ntree = 500
)

# Best parameters and model
print(rf_tuned$bestTune)
rf_best <- rf_tuned$finalModel

# Predictions
rf_pred <- predict(rf_best, newdata = test_data)
rf_metrics <- evaluate_model(test_data$price, rf_pred)
print("Tuned Random Forest Metrics:")
print(rf_metrics)

##XGBoost
param_grid <- expand.grid(max_depth = c(3, 5, 7), eta = c(0.01, 0.1, 0.3), nrounds = c(50, 100, 200))
results <- list()

for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  model <- xgboost(data = dtrain, max_depth = params$max_depth, eta = params$eta,
                   nrounds = params$nrounds, objective = "reg:squarederror", verbose = 0)
  preds <- predict(model, dtest)
  results[[i]] <- evaluate_model(y_test, preds)
}

best_result <- which.min(sapply(results, function(x) x$RMSE))
cat("Best XGBoost Parameters and Metrics:\n")
print(param_grid[best_result, ])
print(results[[best_result]])

##SVR
# Define a grid for tuning
svr_grid <- expand.grid(
  C = c(0.1, 1, 10), 
  epsilon = c(0.01, 0.1, 1),
  gamma = c(0.001, 0.01, 0.1)
)

# Perform grid search
svr_results <- list()
for (i in 1:nrow(svr_grid)) {
  params <- svr_grid[i, ]
  svr_model <- svm(
    price ~ ., 
    data = train_data, 
    kernel = "radial", 
    cost = params$C, 
    epsilon = params$epsilon, 
    gamma = params$gamma
  )
  
  svr_pred <- predict(svr_model, test_data)
  svr_results[[i]] <- list(
    params = params,
    metrics = evaluate_model(test_data$price, svr_pred)
  )
}

# Find the best parameters
best_svr <- which.min(sapply(svr_results, function(x) x$metrics$RMSE))
print("Best SVR Parameters and Metrics:")
print(svr_results[[best_svr]])

