import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Serie = "soybean_data_soilgrid250_modified_states_9"


# Read data from CSV
data = pd.read_csv(f'Datasets/{Serie}.csv')

# Notify if there are missing values before filling with 0
if data.isnull().any().any():
    print("Warning: There are missing values in the dataset before filling with 0.")

# Handle missing values (replace with 0)
data = data.fillna(0)

# Separate the first three columns (to avoid scaling)
first_three_cols = data.iloc[:, :3]  # Select first three columns
remaining_cols = data.iloc[:, 3:]  # Select columns from 4th onwards

# Identify feature groups by prefix
rain_ft = remaining_cols.filter(regex='^W_1_')
solar_ft = remaining_cols.filter(regex='^W_2_')
snwat_ft = remaining_cols.filter(regex='^W_3_')
maxtem_ft = remaining_cols.filter(regex='^W_4_')
mintem_ft = remaining_cols.filter(regex='^W_5_')
vapor_ft = remaining_cols.filter(regex='^W_6_')

bulk_ft = remaining_cols.filter(regex='^bdod')
caex_ft = remaining_cols.filter(regex='^cec')
crfrg_ft = remaining_cols.filter(regex='^cfvo')
clay_ft = remaining_cols.filter(regex='^clay')
nitro_ft = remaining_cols.filter(regex='^nitrogen')
cbdens_ft = remaining_cols.filter(regex='^ocd')
cbstk_ft = remaining_cols.filter(regex='^ocs')
phh2o_ft = remaining_cols.filter(regex='^phh')
sand_ft = remaining_cols.filter(regex='^sand')
silt_ft = remaining_cols.filter(regex='^silt')
soilcb_ft = remaining_cols.filter(regex='^soc')

plant_time_ft = remaining_cols.filter(regex='^P_')


# Compute the mean values for each group to test correlation between groups
group_means = pd.DataFrame({
    'Rain': rain_ft.mean(axis=1),
    'Solar': solar_ft.mean(axis=1),
    'SnWat': snwat_ft.mean(axis=1),
    'MaxTem': maxtem_ft.mean(axis=1),
    'MinTem': mintem_ft.mean(axis=1),
    'Vapor': vapor_ft.mean(axis=1),
    'BulDen': bulk_ft.mean(axis=1),
    'CaEx': caex_ft.mean(axis=1),
    'CrFrg': crfrg_ft.mean(axis=1),
    'Clay': clay_ft.mean(axis=1),
    'Nitro': nitro_ft.mean(axis=1),
    'CbDens': cbdens_ft.mean(axis=1),
    'CbStk': cbstk_ft.mean(axis=1),
    'pHh2o': phh2o_ft.mean(axis=1),
    'Sand': sand_ft.mean(axis=1),
    'Silt': silt_ft.mean(axis=1),
    'SoilCb': soilcb_ft.mean(axis=1),
    'PltTime': plant_time_ft.mean(axis=1)
})

# Compute correlation between groups
group_corr = group_means.corr()

# Plot the correlation matrix between groups
plt.figure(figsize=(12, 8))
sns.heatmap(group_corr, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Between Feature Groups')
plt.show()

# Create data for intermediate CSV (including first three columns)
data_after_imputation = pd.concat([first_three_cols, remaining_cols], axis=1)

# Save data after feature selection (optional)
data_after_imputation.to_csv(f'ProcessedData/after_feature_correlation.csv', index=False)


print("Feature Correlation Testing complete.")
