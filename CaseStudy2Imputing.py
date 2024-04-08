import pandas as pd
import numpy as np

# Loading the Data
file_path1 = "E:/Docs/School/SMU-MSDS/Trimester 4/MSDS-7333 Quantifying the World/Unit 3 Logistic Regression/dataset_diabetes/diabetic_data.csv" 
df = pd.read_csv(file_path1)

# file_path2 = "E:/Docs/School/SMU-MSDS/Trimester 4/MSDS-7333 Quantifying the World/Unit 3 Logistic Regression/dataset_diabetes/IDs_mapping.csv" 
# df = pd.read_csv(file_path2)

# Display data types
print(df.dtypes)
print(df.shape)

## Clean The Data
# Replace '?' with NaN in the entire dataset
df.replace('?', np.nan, inplace=True)

# Identify missing data in each column
missing_data = df.isnull().sum()

# Get columns with non-zero missing values
non_zero_missing_columns = missing_data[missing_data > 0].index
print(non_zero_missing_columns) # race, weight, payer_code, medical_specialty, diag_1, diag2, diag3

# Print counts of each level for features with missing values
for column in non_zero_missing_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

# Imputation method
for column in non_zero_missing_columns:
    if column in ['race', 'weight', 'payer_code', 'medical_specialty']:
        # Convert to string data type for categorical features
        df[column] = df[column].astype(str)
        # Fill NaNs with the mode for categorical features
        mode_value = df[column].mode()[0]  # Assuming the mode is unique
        df[column].fillna(mode_value, inplace=True)
    elif column in ['diag_1', 'diag_2', 'diag_3']:
        # Convert to numeric data type for numerical features
        df[column] = pd.to_numeric(df[column], errors='coerce')
        # Fill NaNs with the mean for numerical features
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

# Display the DataFrame with imputed values
print(df)

df.to_csv("MSDS-7333 Quantifying the World/Unit 3 Logistic Regression/dataset_diabetes/DiabeticDataImp.csv", index=False)

