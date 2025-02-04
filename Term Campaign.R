# ---------------------------------------------
# Step 1: Load Required Libraries
# ---------------------------------------------
library(dplyr)       # Data manipulation
library(ggplot2)     # Data visualization
library(caret)       # Machine learning utilities
library(randomForest)# Random forest model
library(corrplot)    # Correlation plot
library(ROSE)        # Handling imbalanced data
library(pROC)        # ROC curve analysis
library(caTools)     # Train-test split
library(rpart)       # Decision tree model
library(xgboost)     # XGBoost Model
library(e1071)       # Support Vector Machines

# ---------------------------------------------
# Step 2: Load & Inspect the Data
# ---------------------------------------------
bank_data <- read.csv("bank-additional-full.csv")  # Modify if needed
str(bank_data)   # Check structure of data
summary(bank_data)   # Summary statistics

# ---------------------------------------------
# Step 3: Handling Missing Values
# ---------------------------------------------
# Replace missing values with column median (numeric) or mode (categorical)
for (col in colnames(bank_data)) {
  if (is.numeric(bank_data[[col]])) {
    bank_data[[col]][is.na(bank_data[[col]])] <- median(bank_data[[col]], na.rm = TRUE)
  } else {
    bank_data[[col]][is.na(bank_data[[col]])] <- as.character(stats::mode(bank_data[[col]]))
  }
}

# ---------------------------------------------
# Step 4: Encoding Categorical Variables
# ---------------------------------------------
bank_data <- bank_data %>%
  mutate_if(is.character, as.factor)

# ---------------------------------------------
# Step 5: Data Normalization (Scaling)
# ---------------------------------------------
numeric_cols <- sapply(bank_data, is.numeric)
if (sum(numeric_cols) > 0) {
  bank_data[numeric_cols] <- scale(bank_data[numeric_cols])
}

# ---------------------------------------------
# Step 6: Correlation Analysis
# ---------------------------------------------
if (sum(numeric_cols) > 0) {
  correlation_matrix <- cor(bank_data[, numeric_cols], use = "pairwise.complete.obs")
  corrplot(correlation_matrix, method = "circle")
} else {
  print("No numeric columns found in the dataset.")
}

# ---------------------------------------------
# Step 7: Handling Class Imbalance
# ---------------------------------------------
if ("TargetVariable" %in% colnames(bank_data)) {
  bank_data_balanced <- ROSE(TargetVariable ~ ., data = bank_data, seed = 123)$data
}

# ---------------------------------------------
# Step 8: Train-Test Split (80-20)
# ---------------------------------------------
set.seed(123)
split <- sample.split(bank_data$TargetVariable, SplitRatio = 0.8)
train_data <- subset(bank_data, split == TRUE)
test_data <- subset(bank_data, split == FALSE)

# ---------------------------------------------
# Step 9: Train Machine Learning Models
# ---------------------------------------------

# Logistic Regression Model
log_model <- glm(TargetVariable ~ ., data = train_data, family = binomial)

# Decision Tree Model
tree_model <- rpart(TargetVariable ~ ., data = train_data, method="class")

# Random Forest Model
set.seed(123)
rf_model <- randomForest(TargetVariable ~ ., data = train_data, ntree = 100, importance = TRUE)

# ---------------------------------------------
# Step 10: Model Evaluation
# ---------------------------------------------
# Logistic Regression Predictions
log_predictions <- predict(log_model, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)

# Decision Tree Predictions
tree_predictions <- predict(tree_model, test_data, type = "class")

# Random Forest Predictions
rf_predictions <- predict(rf_model, test_data, type = "class")

# Model Performance
confusionMatrix(as.factor(log_pred_class), as.factor(test_data$TargetVariable))
confusionMatrix(as.factor(tree_predictions), as.factor(test_data$TargetVariable))
confusionMatrix(as.factor(rf_predictions), as.factor(test_data$TargetVariable))

# ROC Curve Comparison
roc_log <- roc(test_data$TargetVariable, log_predictions)
roc_rf <- roc(test_data$TargetVariable, as.numeric(rf_predictions))
plot(roc_log, col = "blue", main = "ROC Curves")
lines(roc_rf, col = "red")
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("blue", "red"), lwd = 2)

# ---------------------------------------------
# Step 11: Feature Importance Visualization
# ---------------------------------------------
importance_df <- as.data.frame(importance(rf_model))
ggplot(importance_df, aes(x = reorder(rownames(importance_df), MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Features", y = "Mean Decrease in Gini") +
  theme_minimal()

# ---------------------------------------------
# Step 12: Conclusion & Marketing Strategies
# ---------------------------------------------
cat("
Random Forest is chosen due to high accuracy and AUC.
Adjusting threshold can optimize marketing strategies.
Defensive: Higher threshold, fewer predictions with high probability.
Aggressive: Lower threshold, more predictions with lower probability.
")

# Save Processed Data
write.csv(bank_data, "cleaned_bank_data.csv", row.names = FALSE)

