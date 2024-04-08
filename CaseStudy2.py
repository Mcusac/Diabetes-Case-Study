# Predicting Hospital Readmittance With a Logistic Regression Model

# Importing necessary libraries
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

## Loading the Data
# file path
file_path = "MSDS-7333 Quantifying the World/Unit 3 Logistic Regression/dataset_diabetes/DiabeticDataImp.csv"

df = pd.read_csv(file_path, low_memory=False)

# Display data types
print(df.dtypes)
print(df.shape)

# Separate features (X) and target variable (y)
X = pd.get_dummies(df.drop('readmitted', axis=1), drop_first=True)
print(X.shape)
y = df['readmitted']
print(y.shape)

# Scale the X data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Initialize the logistic regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# Define the hyperparameter grid to search
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': np.logspace(-4, 4, 9),  # Inverse of regularization strength
    'solver': ['liblinear']  # Suitable for small datasets
}

# Create the grid search object
grid_search = GridSearchCV(logistic_regression_model, param_grid, cv=5, scoring='accuracy', n_jobs=-2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set using the best hyperparameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report_output)

# # Train the model on the training data
# logistic_regression_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = logistic_regression_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_report_output = classification_report(y_test, y_pred)

# # Print the evaluation metrics
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:")
# print(classification_report_output)

# print(df['readmitted'].value_counts())

# Get the coefficients and corresponding feature names
coefficients = grid_search.coef_[0]
feature_names = X.columns

# Create a DataFrame to store coefficients and feature names
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by absolute coefficient values in descending order
coefficients_df['AbsoluteCoefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='AbsoluteCoefficient', ascending=False)

# Display the top N important features
top_n_features = 10  # Change this value as needed
print(f"Top {top_n_features} important features:")
print(coefficients_df.head(top_n_features))

