import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

# Including to the path another folder
import sys
sys.path.append(r'ProposedModel')

# Import models
from callbackregression import EarlyStoppingRMSE

# Import dataset from CSV
Data = pd.read_csv('ProcessedData/after_feature_selection.csv')

# Defining the attributes and the target value
X = Data[Data.columns[3:363]].values
y = Data[Data.columns[2]].values

# Reshape y to ensure it is a 2D array with a single column
y = y.reshape(-1, 1)

# split train-test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train on {} samples, test on {} samples, num. of features is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1]
))

# Z-score
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Measure start time 
start_time = time.time() 

# Define TSK model parameters
n_rule = 30  # Num. of rules
lr = 0.005  # learning rate
weight_decay = 1e-8 #L2 regularization
consbn = True
order = 1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = TSK(in_dim=X.shape[1], out_dim=1, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
    else:
        other_param.append(p)
optimizer = AdamW([
    {'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay}
], lr=lr)

# ----------------- split 20% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# ----------------- define the earlystopping callback -----------------
EMSE = EarlyStoppingRMSE(x_val, y_val, verbose=1, patience=100, save_path="tmp.pkl")

wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.MSELoss(),
              epochs=700, callbacks=[EMSE], ur=0, ur_tau=0.5, label_type="r")
wrapper.fit(x_train, y_train)
wrapper.load("tmp.pkl")

y_pred = wrapper.predict(x_test)
# Calculating the error metrics
# Compute the R^2 score
R2 = r2_score(y_test, y_pred)
print("R^2:", R2)
# Compute the Root Mean Square Error
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)

# Measure end time for predictions and error calculation
end_time = time.time()

# Print the timing results at the end of the script
print(f"Running time: {end_time - start_time} sec.")

# Create DataFrame for metrics
metrics_df = pd.DataFrame({
    'n_rule': [n_rule],
    'R2': [R2],
    'RMSE': [RMSE],
    'MAE': [MAE],
    'Running time': [end_time - start_time]
})

# Save metrics to Excel
metrics_file = f'Result/GDmetrics.xlsx'
metrics_df.to_excel(metrics_file, index=False)
print(f"Metrics saved to {metrics_file}")

# Plot the graphic
plt.figure(figsize=(20,12))
plt.rc('font', size=7)
plt.rc('axes', titlesize=7)
plt.plot(y_test, linewidth = 2, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 1, color = 'purple', label = 'TSK-GD')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()