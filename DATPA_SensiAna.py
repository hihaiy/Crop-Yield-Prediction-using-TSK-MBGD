import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.tsk import TSK

from callbackregression import EarlyStoppingRMSE
from trainingregression import Wrapper

# Import dataset from CSV
Data = pd.read_csv('ProcessedData/after_feature_selection.csv')

# Defining the attributes and the target value
X = Data[Data.columns[3:363]].values
y = Data[Data.columns[2]].values

# Reshape y to ensure it is a 2D array with a single column
y = y.reshape(-1, 1)

# Split train-test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Z-score
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Split data for early stopping
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Create DataLoader for mini-batch gradient descent
batch_size = 15
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define TSK model parameters
n_rule = 60  # Num. of rules
lr = 0.005  # learning rate
weight_decay = 1e-8  # L2 regularization
consbn = True
order = 1

# Define antecedent
init_center = antecedent_init_center(x_train.numpy(), y_train.numpy(), n_rule=n_rule)
gmf = nn.Sequential(
    AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
    nn.LayerNorm(n_rule),
    nn.ReLU()
)

# Define full TSK model
model = TSK(in_dim=X.shape[1], out_dim=1, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# Optimizer
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

# Early stopping callback
EMSE = EarlyStoppingRMSE(x_val.numpy(), y_val.numpy(), verbose=1, patience=300, save_path="aaa.pkl")

# Wrapper initialization with mini-batch gradient descent
wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.MSELoss(),
                  epochs=700, callbacks=[EMSE], ur=1, ur_tau=0.5, label_type="r")

# Function to train and evaluate the model
def train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test):
    # Training loop with mini-batch gradient descent
    for epoch in range(700):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Call early stopping evaluation at the end of each epoch
        wrapper.cur_epoch = epoch  # Update the current epoch in the wrapper
        EMSE.on_epoch_end(wrapper)

        if wrapper.stop_training:
            print("Early stopping at epoch {}".format(epoch))
            break

    # Load the best model based on early stopping
    wrapper.load("aaa.pkl")

    # Predictions on test set
    model.eval()
    y_pred = wrapper.predict(x_test)

    # Calculating the error metrics
    R2 = r2_score(y_test.numpy(), y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
    MAE = mean_absolute_error(y_test.numpy(), y_pred)

    return R2, RMSE, MAE

# Perform DA TPA by varying each input parameter
sensitivity_results = []
for i in range(X.shape[1]):
    print(f"Varying feature {i}")

    # Create perturbed datasets by slightly varying the i-th feature
    x_train_perturbed = x_train.clone().detach()
    x_test_perturbed = x_test.clone().detach()

    # Add a small delta to the feature
    delta = 0.1 * x_train[:, i].std().item()
    x_train_perturbed[:, i] += delta
    x_test_perturbed[:, i] += delta

    # Train and evaluate the model on the perturbed data
    R2, RMSE, MAE = train_and_evaluate(x_train_perturbed, y_train, x_val, y_val, x_test_perturbed, y_test)

    # Record the results
    sensitivity_results.append({
        'feature': i,
        'R2': R2,
        'RMSE': RMSE,
        'MAE': MAE
    })

# Convert results to DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results)

# Save sensitivity analysis results to Excel
sensitivity_file = 'Result/sensitivity_analysis.xlsx'
sensitivity_df.to_excel(sensitivity_file, index=False)
print(f"Sensitivity analysis results saved to {sensitivity_file}")

