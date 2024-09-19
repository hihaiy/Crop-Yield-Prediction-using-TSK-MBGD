# Import libraries
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Feature scaling
from sklearn.preprocessing import RobustScaler

# Including to the path another fold
import sys
sys.path.append(r'ProposedModels')

# Import models
from NTSK import NTSK

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "after_feature_selection"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
X = Data[Data.columns[3:363]].values
y = Data[Data.columns[2]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.8)
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Min-max scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# Measure start time 
start_time = time.time()

# ----------------------------------------------------------------------------
# NTSK-RLS
# ----------------------------------------------------------------------------

Model = "NTSK-RLS"

# Set hyperparameters range
n_clusters = 15
lambda1 = 4
RLS_option = 1

# Initialize the model
model = NTSK(n_clusters = n_clusters, lambda1 = lambda1, RLS_option = RLS_option)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred1 = model.predict(X_test)

# Calculating the error metrics
# Compute the number of final rules
Rules = n_clusters
print("Rules:", Rules)
# Compute the R^2 score
R2 = r2_score(y_test, y_pred1)
print("R^2:", R2)
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred1))
print("RMSE:", RMSE)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred1)
print("MAE:", MAE)

# # -------------------------------------------------------------------------
# # NTSK-wRLS
# # # -----------------------------------------------------------------------

Model = "NTSK-wRLS"

# Set hyperparameters range
n_clusters_2 = 85
RLS_option = 2

# Initialize the model
model = NTSK(n_clusters = n_clusters_2, RLS_option = RLS_option)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
y_pred2 = model.predict(X_test)

# Calculating the error metrics
# Compute the number of final rules
Rules = n_clusters_2
print("Rules2:", Rules)
# Compute the R^2 score
R2 = r2_score(y_test, y_pred2)
print("R^2:", R2)
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred2))
print("RMSE2:", RMSE)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred2)
print("MAE2:", MAE)

# Measure end time for predictions and error calculation
end_time = time.time()

# Print the timing results at the end of the script
print(f"Running time: {end_time - start_time} sec.")

# Plot the graphic
plt.figure(figsize=(20,12))
plt.rc('font', size=7)
plt.rc('axes', titlesize=7)
plt.plot(y_test, linewidth = 2, color = 'red', label = 'Actual value')
plt.plot(y_pred1, linewidth = 1, color = 'green', label = 'NTSK-RLS', linestyle = "--")
plt.plot(y_pred2, linewidth = 1, color = 'black', label = 'NTSK-wRLS', linestyle = "-.")
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()
