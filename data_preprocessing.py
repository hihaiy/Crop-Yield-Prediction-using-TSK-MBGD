import pandas as pd

Serie = "soybean_data_soilgrid250_modified_states_9"

"""
Preprocesses data from a CSV file, handling missing values, outliers (based on variance), and standardizing features.

Args:
    data_file (str): Path to the CSV file.

Returns:
    pandas.DataFrame: Preprocessed data with missing values filled, outliers removed,
                        and features standardized.
"""

# Read data from CSV
data = pd.read_csv(f'Datasets/{Serie}.csv')


# Notify if there are missing values before filling with 0
if data.isnull().any().any():
    print("Warning: There are missing values in the dataset before filling with 0.")

# Handle missing values (replace with 0)
data = data.fillna(0)

# Remove columns with all data equal to 0
data = data.loc[:, (data != 0).any(axis=0)]

# Filter data to include only years 2016, 2017, and 2018
data = data[(data['year'] >= 2016) & (data['year'] <= 2018)]

# Sort data by 'year' column in ascending order
data = data.sort_values(by='year', ascending=True)


# Separate the first two columns (to avoid scaling)
first_three_cols = data.iloc[:, :3]  # Select first three columns
remaining_cols = data.iloc[:, 3:]  # Select columns from 4th onwards


# Analyze variance and identify features to exclude (assuming 95% threshold)
variance_threshold = 0.95
high_variance_features = remaining_cols.var() > remaining_cols.var().quantile(variance_threshold)
features_to_keep = remaining_cols.columns[~high_variance_features]
remaining_cols = remaining_cols[features_to_keep]  # Keep only features below the variance threshold


# Create data for intermediate CSV (including first three columns)
data_after_imputation = pd.concat([first_three_cols, remaining_cols], axis=1)

# Save data after feature selection (optional)
data_after_imputation.to_csv(f'ProcessedData/after_feature_selection.csv', index=False)
  

print("Preprocessing complete. Final processed data returned.")
