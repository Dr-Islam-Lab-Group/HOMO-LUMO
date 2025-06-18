import warnings
warnings.filterwarnings("ignore")

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

#from matplotlib import pyplot as plt
#import matplotlib.patches as mpatches
#import seaborn as sn

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

# Output_directory
fd = "Output"

# Load training, validation, and test datasets
train_file = "dataset/GDB13_subset/split/train.csv"
val_file = "dataset/GDB13_subset/split/valid.csv"
test_file = "dataset/GDB13_subset/split/test.csv"

df_train = pd.read_csv(train_file)
df_val = pd.read_csv(val_file)
df_test = pd.read_csv(test_file)

# Combine datasets for descriptor generation and feature selection
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
new_data = df_all[['smiles', 'homolumogap']]

# Generate RDKit descriptors
def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    Mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)  # Add hydrogens
        descriptors = calc.CalcDescriptors(mol)  # Calculate descriptors
        Mol_descriptors.append(descriptors)

    return Mol_descriptors, desc_names

Mol_descriptors, desc_names = RDkit_descriptors(new_data['smiles'])

# Create a DataFrame for descriptors
df_descriptor = pd.DataFrame(Mol_descriptors, columns=desc_names)

# Remove correlated features across the whole dataset
def remove_correlated_features(descriptors):
    correlated_matrix = descriptors.corr().abs()
    upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.90)]
    print(f"Removing correlated features: {to_drop}")
    return descriptors.drop(columns=to_drop, axis=1)

df_descriptor_filtered = remove_correlated_features(df_descriptor)

# Re-split into respective datasets
X_all = df_descriptor_filtered
y_all = new_data['homolumogap']

# Retrieve indices corresponding to original splits
n_train = len(df_train)
n_val = len(df_val)

X_train, y_train = X_all.iloc[:n_train], y_all.iloc[:n_train]
X_val, y_val = X_all.iloc[n_train:n_train+n_val], y_all.iloc[n_train:n_train+n_val]
X_test, y_test = X_all.iloc[n_train+n_val:], y_all.iloc[n_train+n_val:]

# Verify dataset shapes
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

"""# Catboost Regressor"""

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Initialize the CatBoostRegressor with the best hyperparameters
final_model_CatB = CatBoostRegressor(
    iterations=4000,
    depth=10,
    learning_rate=0.08003866739082473,
    l2_leaf_reg=1.2219300331335978,
    bagging_temperature=0.9026337873010564,
    random_seed=101,
    verbose=100,  # Log progress every 100 iterations
    eval_metric='MAE',  # Use MAE as evaluation metric
    loss_function='RMSE'  # Use RMSE as the loss function
)

# Train the model with the validation set
final_model_CatB.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),  # Validation data for monitoring
    early_stopping_rounds=100,  # Stop training if no improvement for 100 iterations
    use_best_model=True        # Retain the best model during training
)

# Retrieve training and validation losses
evals_result = final_model_CatB.get_evals_result()

# Extract training and validation RMSE losses
train_loss_rmse = evals_result['learn']['RMSE']  # 'RMSE' for training loss
val_loss_rmse = evals_result['validation']['RMSE']  # 'RMSE' for validation loss

# Convert RMSE to MSE (square the values)
train_loss_mse = [loss**2 for loss in train_loss_rmse]
val_loss_mse = [loss**2 for loss in val_loss_rmse]

# Save the MSE losses to a CSV file
loss_df = pd.DataFrame({
    'Iteration': range(1, len(train_loss_mse) + 1),
    'Training Loss (MSE)': train_loss_mse,
    'Validation Loss (MSE)': val_loss_mse
})

loss_df.to_csv(f'{fd}/catboost_train_validation_losses.csv', index=False)

# Plot the training and validation losses
import pandas as pd
import matplotlib.pyplot as plt

# Load the losses from the CSV file
loss_df = pd.read_csv(f'{fd}/catboost_train_validation_losses.csv')

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(loss_df['Iteration'], loss_df['Training Loss (MSE)'], label='Training Loss', color='blue')
plt.plot(loss_df['Iteration'], loss_df['Validation Loss (MSE)'], label='Validation Loss', color='red')
plt.xlabel('Iterations',fontsize=22, fontweight='bold')
plt.ylabel('Loss (eV)', fontsize=22, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
#plt.title(f'{fd}/catboost_train_validation_losses.csv')
plt.legend()
plt.grid(True)
plt.savefig(f'{fd}/catboost_model_train_validation_losses.png', dpi=300, bbox_inches='tight')
#plt.show()

# save the trained model weight
final_model_CatB.save_model(f'{fd}/CatBoost_model.weights.h5')

#load saved model
final_model_CatB.load_model(f'{fd}/CatBoost_model.weights.h5')

# Predict on the testing data
predictions_val_catB = final_model_CatB.predict(X_val)
predictions_test_catB = final_model_CatB.predict(X_test)
predictions_train_catB = final_model_CatB.predict(X_train)

# Calculate MAE
mae_test_catB = mean_absolute_error(y_test, predictions_test_catB)
mae_val_catB = mean_absolute_error(y_val, predictions_val_catB)
mae_train_catB = mean_absolute_error(y_train, predictions_train_catB)
print("Mean Absolute Error:", mae_test_catB)
print("Mean Absolute Error:", mae_val_catB)
print("Mean Absolute Error:", mae_train_catB)

# R^2 (coefficient of determination) regression score function:
R2_test_catB =r2_score(y_test, predictions_test_catB)
R2_val_catB =r2_score(y_val, predictions_val_catB)
R2_train_catB =r2_score(y_train, predictions_train_catB)
print('R^2:', R2_test_catB)
print('R^2:', R2_val_catB)
print('R^2:', R2_train_catB)


#MSE
from sklearn.metrics import mean_squared_error
mse_test_catB = mean_squared_error(y_test, predictions_test_catB)
mse_val_catB = mean_squared_error(y_val, predictions_val_catB)
mse_train_catB = mean_squared_error(y_train, predictions_train_catB)
print('MSE:', mse_test_catB)
print('MSE:', mse_val_catB)
print('MSE:', mse_train_catB)

#RMSE
from sklearn.metrics import mean_squared_error
rmse_test_catB = mean_squared_error(y_test, predictions_test_catB)
rmse_val_catB = mean_squared_error(y_val, predictions_val_catB)
rmse_train_catB = mean_squared_error(y_train, predictions_train_catB)
print('RMSE:', rmse_test_catB)
print('RMSE:', rmse_val_catB)
print('RMSE:', rmse_train_catB)

#Save the metrics in a csv file
import pandas as pd
df_test = pd.DataFrame({'MAE': [mae_test_catB], 'R2': [R2_test_catB], 'MSE': [mse_test_catB], 'RMSE': [rmse_test_catB]})
df_val = pd.DataFrame({'MAE': [mae_val_catB], 'R2': [R2_val_catB], 'MSE': [mse_val_catB], 'RMSE': [rmse_val_catB]})
df_train = pd.DataFrame({'MAE': [mae_train_catB], 'R2': [R2_train_catB], 'MSE': [mse_train_catB], 'RMSE': [rmse_train_catB]})


#Save df in csv file
df_test.to_csv(f'{fd}/CatB_model_test_metrics.csv', index=False)
df_val.to_csv(f'{fd}/CatB_model_val_metrics.csv', index=False)
df_train.to_csv(f'{fd}/CatB_model_train_metrics.csv', index=False)

import pandas as pd

# Create DataFrame for actual vs predicted values for test set
df_test_predictions = pd.DataFrame({
    'Actual': y_test,  # Actual values
    'Predicted': predictions_test_catB  # Predicted values
})

# Create DataFrame for actual vs predicted values for val set
df_val_predictions = pd.DataFrame({
    'Actual': y_val,  # Actual values
    'Predicted': predictions_val_catB  # Predicted values
})

# Create DataFrame for actual vs predicted values for train set
df_train_predictions = pd.DataFrame({
    'Actual': y_train,  # Actual values
    'Predicted': predictions_train_catB  # Predicted values
})


# Save the DataFrames to CSV
df_test_predictions.to_csv(f'{fd}/CatB_model_test_actual_vs_predicted.csv', index=False)
df_val_predictions.to_csv(f'{fd}/CatB_model_val_actual_vs_predicted.csv', index=False)
df_train_predictions.to_csv(f'{fd}/CatB_model_train_actual_vs_predicted.csv', index=False)

print("CSV files saved successfully!")

#load the csv file
test_plot=pd.read_csv(f'{fd}/CatB_model_test_actual_vs_predicted.csv')
val_plot = pd.read_csv(f'{fd}/CatB_model_val_actual_vs_predicted.csv')
train_plot = pd.read_csv(f'{fd}/CatB_model_train_actual_vs_predicted.csv')

#load the csv file
test_plot=pd.read_csv(f'{fd}/CatB_model_test_actual_vs_predicted.csv')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = test_plot['Actual']
predicted = test_plot['Predicted']

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)


fig, ax = plt.subplots(figsize=(10, 6))

hb1 = ax.hist2d(actual, predicted, bins=150, norm=colors.LogNorm(), cmap='plasma')
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
        color='red', linestyle='--', label='Ideal Prediction')
ax.set_xlabel('Actual Value', fontsize=22, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=22, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

#ax.set_title('Actual vs. Predicted Values', fontsize=16
#ax.text(0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')

slope, intercept, r_value, _, _ = linregress(actual, predicted)
ax.text(0.05, 0.85, f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}', transform=ax.transAxes,
        fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


ax.text(0.60, 0.1, f'MAE = {mae:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.2, f'RMSE = {rmse:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.3, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cb = plt.colorbar(hb1[3], ax=ax, label='Count')
cb.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
cb.set_label('Count', fontsize=18, fontweight='bold')  # Adjust the font size of the colorbar label

plt.savefig(f'{fd}/catB_actual_vs_predicted_test.png', dpi=300, bbox_inches='tight')
#plt.show()

"""# Optimized LGBMRegressor"""

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import os

# Assuming X_train, X_val, X_test, y_train, y_val, y_test are already defined

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the LightGBM model with MSE as loss
lgbm_model = LGBMRegressor(
    force_col_wise=True,
    learning_rate=0.03,
    num_leaves=125,
    max_depth=20,
    min_child_samples=15,
    subsample=1.0,
    colsample_bytree=0.7,
    n_estimators=4000,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=101,
    verbose=100,
    objective='regression',  # Default for regression (MSE loss)
)

# Train the model with MAE as the evaluation metric
lgbm_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
    eval_metric=['l1', 'l2'],  # Set MAE ('l1') and MSE ('l2') as the evaluation metric
    callbacks=[lgb.early_stopping(stopping_rounds=100)]  # Early stopping
)

# Retrieve training and validation losses (MAE)
evals_result = lgbm_model.evals_result_

# Extract MAE losses (the evaluation metric)
train_mae = evals_result['training']['l1'] # 'l1' is the key for MAE
val_mae = evals_result['valid_1']['l1'] # 'l1' is the key for MAE

# Extract the MSE from the model's default loss (this is the loss function used during training)
train_mse = evals_result['training']['l2']
val_mse = evals_result['valid_1']['l2']

# Get the minimum length among all loss lists
min_len = min(len(train_mse), len(val_mse), len(train_mae), len(val_mae))

# Save the losses to a CSV file
loss_df = pd.DataFrame({
    'Iteration': range(1, min_len + 1),
    'Training MSE': train_mse[:min_len],
    'Validation MSE': val_mse[:min_len],
    'Training MAE': train_mae[:min_len],
    'Validation MAE': val_mae[:min_len]
})

# Save the DataFrame to a CSV file
loss_df.to_csv(f'{fd}/lgbm_train_validation_losses.csv', index=False)

# Plot the training and validation losses
import pandas as pd
import matplotlib.pyplot as plt

# Load the losses from the CSV file
loss_df = pd.read_csv(f'{fd}/lgbm_train_validation_losses.csv')

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(loss_df['Iteration'], loss_df['Training MSE'], label='Training Loss', color='blue')
plt.plot(loss_df['Iteration'], loss_df['Validation MSE'], label='Validation Loss', color='red')
plt.xlabel('Iteration',fontsize=22, fontweight='bold')
plt.ylabel('Loss (eV)', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

#plt.title(f'{fd}/lgbm_train_validation_losses.csv', fontsize=18, fontweight='bold')
plt.legend()
plt.grid(True)
plt.savefig(f'{fd}/lgbm_model_train_validation_losses.png', dpi=300, bbox_inches='tight')
#plt.show()

#save trained lgbm model so that I can load it latter
import pickle
filename = f'{fd}/LGBM_model_weights.txt'
pickle.dump(lgbm_model, open(filename, 'wb'))

#load saved model
import pickle
filename = f'{fd}/LGBM_model_weights.txt'
lgbm_model = pickle.load(open(filename, 'rb'))

# Predict on the testing data
predictions_val_lgbm = lgbm_model.predict(X_val_scaled)
predictions_test_lgbm = lgbm_model.predict(X_test_scaled)
predictions_train_lgbm = lgbm_model.predict(X_train_scaled)

# Calculate MAE
mae_test_lgbm = mean_absolute_error(y_test, predictions_test_lgbm)
mae_val_lgbm = mean_absolute_error(y_val, predictions_val_lgbm)
mae_train_lgbm = mean_absolute_error(y_train, predictions_train_lgbm)
print("Mean Absolute Error:", mae_test_lgbm)
print("Mean Absolute Error:", mae_val_lgbm)
print("Mean Absolute Error:", mae_train_lgbm)

# R^2 (coefficient of determination) regression score function:
R2_test =r2_score(y_test, predictions_test_lgbm)
R2_val =r2_score(y_val, predictions_val_lgbm)
R2_train =r2_score(y_train, predictions_train_lgbm)
print('R^2:', R2_test)
print('R^2:', R2_val)
print('R^2:', R2_train)

#MSE
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, predictions_test_lgbm)
mse_val = mean_squared_error(y_val, predictions_val_lgbm)
mse_train = mean_squared_error(y_train, predictions_train_lgbm)
print('MSE:', mse_test)
print('MSE:', mse_val)
print('MSE:', mse_train)

#RMSE
rmse_test = mean_squared_error(y_test, predictions_test_lgbm)
rmse_val = mean_squared_error(y_val, predictions_val_lgbm)
rmse_train = mean_squared_error(y_train, predictions_train_lgbm)
print('RMSE:', rmse_test)

#Save the metrics in a csv file
import pandas as pd
df_test = pd.DataFrame({'MAE': [mae_test_lgbm], 'R2': [R2_test], 'MSE': [mse_test], 'RMSE': [rmse_test]})
df_val = pd.DataFrame({'MAE': [mae_val_lgbm], 'R2': [R2_val], 'MSE': [mse_val], 'RMSE': [rmse_val]})
df_train = pd.DataFrame({'MAE': [mae_train_lgbm], 'R2': [R2_train], 'MSE': [mse_train], 'RMSE': [rmse_train]})
#Save df in csv file
df_test.to_csv(f'{fd}/LGBM_model_test_metrics.csv', index=False)
df_val.to_csv(f'{fd}/LGBM_model_val_metrics.csv', index=False)
df_train.to_csv(f'{fd}/LGBM_model_train_metrics.csv', index=False)

import pandas as pd

# Create DataFrame for actual vs predicted values for test set
df_test_predictions = pd.DataFrame({
    'Actual': y_test,  # Actual values
    'Predicted': predictions_test_lgbm  # Predicted values
})
df_val_predictions = pd.DataFrame({
    'Actual': y_val,  # Actual values
    'Predicted': predictions_val_lgbm  # Predicted values
})
df_train_predictions = pd.DataFrame({
    'Actual': y_train,  # Actual values
    'Predicted': predictions_train_lgbm  # Predicted values
})


# Save the DataFrames to CSV
df_test_predictions.to_csv(f'{fd}/lgbm_test_actual_vs_predicted.csv', index=False)
df_val_predictions.to_csv(f'{fd}/lgbm_val_actual_vs_predicted.csv', index=False)
df_train_predictions.to_csv(f'{fd}/lgbm_train_actual_vs_predicted.csv', index=False)

print("CSV files saved successfully!")

#load the csv file
lgbmR_plot=pd.read_csv(f'{fd}/lgbm_test_actual_vs_predicted.csv')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = lgbmR_plot['Actual']
predicted = lgbmR_plot['Predicted']

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)


fig, ax = plt.subplots(figsize=(10, 6))

hb1 = ax.hist2d(actual, predicted, bins=150, norm=colors.LogNorm(), cmap='plasma')
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
        color='red', linestyle='--', label='Ideal Prediction')
ax.set_xlabel('Actual Value', fontsize=22, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=22, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

#ax.set_title('Actual vs. Predicted Values', fontsize=16
#ax.text(0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')

slope, intercept, r_value, _, _ = linregress(actual, predicted)
ax.text(0.05, 0.85, f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}', transform=ax.transAxes,
        fontsize=16, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


ax.text(0.60, 0.1, f'MAE = {mae:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.2, f'RMSE = {rmse:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.3, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cb = plt.colorbar(hb1[3], ax=ax, label='Count')
cb.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
cb.set_label('Count', fontsize=16, fontweight='bold')  # Adjust the font size of the colorbar label

plt.savefig(f'{fd}/lgbmR_actual_vs_predicted_test.png', dpi=300, bbox_inches='tight')
#plt.show()

"""# Bidirectional LSTM"""

X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])

X_test_reshaped.shape

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import random as random

# Set random seed for reproducibility
seed_value = 101
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM (3D input: samples, timesteps, features)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = np.reshape(X_val_scaled, (X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the BiLSTM model
Bilstm_model = Sequential([
    Bidirectional(LSTM(units=64, activation='relu', return_sequences=True), input_shape=(1, X_train.shape[1])),
    Bidirectional(LSTM(units=64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(units=64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(units=64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(units=64, activation='relu')),
    Dense(units=1)  # Output layer for regression
])

# Define a custom learning rate for Adam optimizer
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)  # Custom learning rate

# Compile the model with Mean Squared Error loss function
Bilstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
history = Bilstm_model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=1000,
    batch_size=128,
    verbose=1,
    callbacks=[early_stopping]
)

# Extract MSE losses
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Save the MSE losses to a DataFrame
loss_df = pd.DataFrame({
    'Epoch': range(1, len(training_loss) + 1),
    'Training MSE': training_loss,
    'Validation MSE': validation_loss
})

# Save losses to a CSV file
loss_df.to_csv(f'{fd}/bilstm_train_validation_losses.csv', index=False)

# Plotting the training and validation loss
import pandas as pd
import matplotlib.pyplot as plt

# Load the losses from the CSV file
loss_df = pd.read_csv(f'{fd}/bilstm_train_validation_losses.csv')
# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(loss_df['Epoch'], loss_df['Training MSE'], label='Training Loss')
plt.plot(loss_df['Epoch'], loss_df['Validation MSE'], label='Validation Loss')
plt.xlabel('Epochs',fontsize=22, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
#plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(f'{fd}/bilstm_model_train_validation_losses.png', dpi=300, bbox_inches='tight')
#plt.show()

#save the model weight
Bilstm_model.save_weights(f'{fd}/Bi_LSTM_model.weights.h5')

#load the saved model
Bilstm_model.load_weights(f'{fd}/Bi_LSTM_model.weights.h5')

# Predict on the testing data
predictions_test_Bilstm = Bilstm_model.predict(X_test_reshaped)
predictions_val_Bilstm = Bilstm_model.predict(X_val_reshaped)
predictions_train_Bilstm = Bilstm_model.predict(X_train_reshaped)

# Calculate MAE
mae_test = mean_absolute_error(y_test, predictions_test_Bilstm)
mae_val = mean_absolute_error(y_val, predictions_val_Bilstm)
mae_train = mean_absolute_error(y_train, predictions_train_Bilstm)
print("Mean Absolute Error:", mae_test)
print("Mean Absolute Error:", mae_val)
print("Mean Absolute Error:", mae_train)

# R^2 (coefficient of determination) regression score function:
R2_test =r2_score(y_test, predictions_test_Bilstm)
R2_val =r2_score(y_val, predictions_val_Bilstm)
R2_train =r2_score(y_train, predictions_train_Bilstm)
print('R^2:', R2_test)
print('R^2:', R2_val)
print('R^2:', R2_train)

#MSE
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, predictions_test_Bilstm)
mse_val = mean_squared_error(y_val, predictions_val_Bilstm)
mse_train = mean_squared_error(y_train, predictions_train_Bilstm)
print('MSE:', mse_test)
print('MSE:', mse_val)
print('MSE:', mse_train)

#RMSE
rmse_test = mean_squared_error(y_test, predictions_test_Bilstm)
rmse_val = mean_squared_error(y_val, predictions_val_Bilstm)
rmse_train = mean_squared_error(y_train, predictions_train_Bilstm)
print('RMSE:', rmse_test)

#Save the metrics in a csv file
import pandas as pd
df_test = pd.DataFrame({'MAE': [mae_test], 'R2': [R2_test], 'MSE': [mse_test], 'RMSE': [rmse_test]})
df_val = pd.DataFrame({'MAE': [mae_val], 'R2': [R2_val], 'MSE': [mse_val], 'RMSE': [rmse_val]})
df_train = pd.DataFrame({'MAE': [mae_train], 'R2': [R2_train], 'MSE': [mse_train], 'RMSE': [rmse_train]})
#Save df in csv file
df_test.to_csv(f'{fd}/Bilstm_model_test_metrics.csv', index=False)
df_val.to_csv(f'{fd}/Bilstm_model_val_metrics.csv', index=False)
df_train.to_csv(f'{fd}/Bilstm_model_train_metrics.csv', index=False)

import pandas as pd

# Create DataFrame for actual vs predicted values for test set
df_test_predictions = pd.DataFrame({
    'Actual': y_test,  # Actual values
    'Predicted': predictions_test_Bilstm.flatten()   # Predicted values
})
df_val_predictions = pd.DataFrame({
    'Actual': y_val,  # Actual values
    'Predicted': predictions_val_Bilstm.flatten()   # Predicted values
})
df_train_predictions = pd.DataFrame({
    'Actual': y_train,  # Actual values
    'Predicted': predictions_train_Bilstm.flatten()   # Predicted values
})

# Save the DataFrames to CSV
df_test_predictions.to_csv(f'{fd}/Bilstm_test_actual_vs_predicted.csv', index=False)
df_val_predictions.to_csv(f'{fd}/Bilstm_val_actual_vs_predicted.csv', index=False)
df_train_predictions.to_csv(f'{fd}/Bilstm_train_actual_vs_predicted.csv', index=False)

print("CSV files saved successfully!")

#load the csv file
Bilstm_plot=pd.read_csv(f'{fd}/Bilstm_test_actual_vs_predicted.csv')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = Bilstm_plot['Actual']
predicted = Bilstm_plot['Predicted']

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)


fig, ax = plt.subplots(figsize=(10, 6))

hb1 = ax.hist2d(actual, predicted, bins=150, norm=colors.LogNorm(), cmap='plasma')
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
        color='red', linestyle='--', label='Ideal Prediction')
ax.set_xlabel('Actual Value', fontsize=22, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=22, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

#ax.set_title('Actual vs. Predicted Values', fontsize=16
#ax.text(0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')

slope, intercept, r_value, _, _ = linregress(actual, predicted)
ax.text(0.05, 0.85, f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}', transform=ax.transAxes,
        fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


ax.text(0.60, 0.1, f'MAE = {mae:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.2, f'RMSE = {rmse:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.3, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cb = plt.colorbar(hb1[3], ax=ax, label='Count')
cb.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
cb.set_label('Count', fontsize=16, fontweight='bold')  # Adjust the font size of the colorbar label

plt.savefig(f'{fd}/Bilstm_actual_vs_predicted_test.png', dpi=300, bbox_inches='tight')
#plt.show()

"""# Multilayer Perceptron (MLP)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Assuming X_train, X_val, X_test, y_train, y_val, y_test are already defined

# Scale the data (train, validation, and test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPRegressor
mlp1 = MLPRegressor(
    hidden_layer_sizes=(500, 100, 50),
    max_iter=1000,
    activation='relu',
    solver='adam',
    random_state=101,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    warm_start=True
)

# Initialize lists to store losses
train_losses = []
val_losses = []
epochs = []

# Early stopping parameters
patience = 100
best_val_loss = float('inf')
epochs_without_improvement = 0

# Train the model with manual epoch loop
for epoch in range(1000):  # You can set a higher number of epochs
    mlp1.fit(X_train_scaled, y_train)

    # Compute MSE for training loss (MLP already uses squared_loss by default)
    train_loss = mlp1.loss_  # MLP uses squared_loss (MSE)
    train_losses.append(train_loss)

    # Compute MSE for validation loss
    val_pred = mlp1.predict(X_val_scaled)
    val_loss = mean_squared_error(y_val, val_pred)
    val_losses.append(val_loss)

    # Store epoch number
    epochs.append(epoch + 1)

    print(f"Epoch {epoch + 1} - Training MSE: {train_loss:.4f}, Validation MSE: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
        break

# Save MSE losses to CSV
loss_df = pd.DataFrame({
    'Epoch': epochs,
    'Training MSE': train_losses,
    'Validation MSE': val_losses
})

# Save the DataFrame to CSV
loss_df.to_csv(f'{fd}/mlp1_train_validation_losses.csv', index=False)

import pickle

# Save the entire model using pickle
with open(f'{fd}/mlp1_model.pkl', 'wb') as file:
    pickle.dump(mlp1, file)

#Load the saved model
with open(f'{fd}/mlp1_model.pkl', 'rb') as file:
    mlp1 = pickle.load(file)

# Predict on the testing data
predictions_test_mlp1 = mlp1.predict(X_test_scaled)
predictions_val_mlp1 = mlp1.predict(X_val_scaled)
predictions_train_mlp1 = mlp1.predict(X_train_scaled)

# Calculate MAE
mae_test = mean_absolute_error(y_test, predictions_test_mlp1)
mae_val = mean_absolute_error(y_val, predictions_val_mlp1)
mae_train = mean_absolute_error(y_train, predictions_train_mlp1)
print("Mean Absolute Error:", mae_test)
print("Mean Absolute Error:", mae_val)
print("Mean Absolute Error:", mae_train)

# R^2 (coefficient of determination) regression score function:
R2_test =r2_score(y_test, predictions_test_mlp1)
R2_val =r2_score(y_val, predictions_val_mlp1)
R2_train =r2_score(y_train, predictions_train_mlp1)
print('R^2:', R2_test)
print('R^2:', R2_val)
print('R^2:', R2_train)

#MSE
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, predictions_test_mlp1)
mse_val = mean_squared_error(y_val, predictions_val_mlp1)
mse_train = mean_squared_error(y_train, predictions_train_mlp1)
print('MSE:', mse_test)
print('MSE:', mse_val)
print('MSE:', mse_train)

#RMSE
rmse_test = mean_squared_error(y_test, predictions_test_mlp1)
rmse_val = mean_squared_error(y_val, predictions_val_mlp1)
rmse_train = mean_squared_error(y_train, predictions_train_mlp1)
print('RMSE:', rmse_test)

#Save the metrics in a csv file
import pandas as pd
df_test = pd.DataFrame({'MAE': [mae_test], 'R2': [R2_test], 'MSE': [mse_test], 'RMSE': [rmse_test]})
df_val = pd.DataFrame({'MAE': [mae_val], 'R2': [R2_val], 'MSE': [mse_val], 'RMSE': [rmse_val]})
df_train = pd.DataFrame({'MAE': [mae_train], 'R2': [R2_train], 'MSE': [mse_train], 'RMSE': [rmse_train]})
#Save df in csv file
df_test.to_csv(f'{fd}/mlp1_model_test_metrics.csv', index=False)
df_val.to_csv(f'{fd}/mlp1_model_val_metrics.csv', index=False)
df_train.to_csv(f'{fd}/mlp1_model_train_metrics.csv', index=False)

import pandas as pd

# Create DataFrame for actual vs predicted values for test set
df_test_predictions = pd.DataFrame({
    'Actual': y_test,  # Actual values
    'Predicted': predictions_test_mlp1  # Predicted values
})

df_val_predictions = pd.DataFrame({
    'Actual': y_val,  # Actual values
    'Predicted': predictions_val_mlp1  # Predicted values
})

df_train_predictions = pd.DataFrame({
    'Actual': y_train,  # Actual values
    'Predicted': predictions_train_mlp1  # Predicted values
})

# Save the DataFrames to CSV
df_test_predictions.to_csv(f'{fd}/mlp1_test_actual_vs_predicted.csv', index=False)
df_val_predictions.to_csv(f'{fd}/mlp1_val_actual_vs_predicted.csv', index=False)
df_train_predictions.to_csv(f'{fd}/mlp1_train_actual_vs_predicted.csv', index=False)

print("CSV files saved successfully!")

# Plotting the training and validation loss
loss_df = pd.read_csv(f'{fd}/mlp1_train_validation_losses.csv')

plt.figure(figsize=(10, 6))
plt.plot(loss_df['Epoch'], loss_df['Training MSE'], label='Training Loss', color='blue')
plt.plot(loss_df['Epoch'], loss_df['Validation MSE'], label='Validation Loss', color='red')
plt.xlabel('Epochs',fontsize=22, fontweight='bold')
plt.ylabel('Loss', fontsize=22, fontweight='bold')
plt.tick_params(axis='x', labelsize=18) # Changed to plt.tick_params and specified axis
plt.tick_params(axis='y', labelsize=18)
plt.legend()
plt.grid(True)
plt.savefig(f'{fd}/mlp1_train_validation_losses.png', dpi=300, bbox_inches='tight')
#plt.show()

#load the csv file
mlp1_plot=pd.read_csv(f'{fd}/mlp1_test_actual_vs_predicted.csv')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = mlp1_plot['Actual']
predicted = mlp1_plot['Predicted']

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)


fig, ax = plt.subplots(figsize=(10, 6))

hb1 = ax.hist2d(actual, predicted, bins=150, norm=colors.LogNorm(), cmap='plasma')
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
        color='red', linestyle='--', label='Ideal Prediction')
ax.set_xlabel('Actual Value', fontsize=22, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=22, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

#ax.set_title('Actual vs. Predicted Values', fontsize=16
#ax.text(0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')

slope, intercept, r_value, _, _ = linregress(actual, predicted)
ax.text(0.05, 0.85, f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}', transform=ax.transAxes,
        fontsize=16, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


ax.text(0.60, 0.1, f'MAE = {mae:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.2, f'RMSE = {rmse:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.3, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cb = plt.colorbar(hb1[3], ax=ax, label='Count')
cb.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
cb.set_label('Count', fontsize=16, fontweight='bold')  # Adjust the font size of the colorbar label

plt.savefig(f'{fd}/mlp1_actual_vs_predicted_test.png', dpi=300, bbox_inches='tight')
#plt.show()

"""# Ensemble of LGBMRegressor, LSTM, MLP and CatboostRegressor"""

# Generate predictions on the training, Validation and testing data
lgbm_train_preds = lgbm_model.predict(X_train_scaled)
lstm_train_preds = Bilstm_model.predict(X_train_reshaped).flatten()
mlp_train_preds = mlp1.predict(X_train_scaled)
catb_train_preds = final_model_CatB.predict(X_train)

lgbm_val_preds = lgbm_model.predict(X_val_scaled)
lstm_val_preds = Bilstm_model.predict(X_val_reshaped).flatten()
mlp_val_preds = mlp1.predict(X_val_scaled)
catb_val_preds = final_model_CatB.predict(X_val)

lgbm_test_preds = lgbm_model.predict(X_test_scaled)
lstm_test_preds = Bilstm_model.predict(X_test_reshaped).flatten()
mlp_test_preds = mlp1.predict(X_test_scaled)
catb_test_preds = final_model_CatB.predict(X_test)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Define weights for each base model
lgbm_weight = 0.70
lstm_weight = 0.20
mlp_weight = 0.10
#cat_weight = 0.00

# Combine predictions with weights for training and validation
weighted_train_preds = (
    lgbm_weight * lgbm_train_preds +
    lstm_weight * lstm_train_preds +
    mlp_weight * mlp_train_preds
    #cat_weight * catb_train_preds
)

weighted_val_preds = (
    lgbm_weight * lgbm_val_preds +
    lstm_weight * lstm_val_preds +
    mlp_weight * mlp_val_preds
    #cat_weight * catb_val_preds
)

weighted_test_preds = (
    lgbm_weight * lgbm_test_preds +
    lstm_weight * lstm_test_preds +
    mlp_weight * mlp_test_preds
    #cat_weight * catb_test_preds
)

# Define a random state for reproducibility
random_state = 101

# Initialize the Gradient Boosting model with MSE as loss function
esm_model = GradientBoostingRegressor(
    n_estimators=4000,
    learning_rate=0.001,
    max_depth=10,
    loss='squared_error',  # Explicitly setting MSE as the loss function
    random_state=random_state  # Ensuring reproducibility
)

# Fit the model
esm_model.fit(weighted_train_preds.reshape(-1, 1), y_train)

# Initialize lists to store MSE losses
training_losses = []
validation_losses = []

# Compute MSE at each iteration using staged_predict
for i, y_pred in enumerate(esm_model.staged_predict(weighted_train_preds.reshape(-1, 1))):
    # Compute MSE for training
    train_loss = mean_squared_error(y_train, y_pred)

    # Predict on validation data using the current stage
    val_pred = list(esm_model.staged_predict(weighted_val_preds.reshape(-1, 1)))[i]
    val_loss = mean_squared_error(y_val, val_pred)

    # Save the MSE losses
    training_losses.append(train_loss)
    validation_losses.append(val_loss)



# Save losses to CSV
loss_df = pd.DataFrame({
    'Iteration': range(1, len(training_losses) + 1),
    'Training MSE': training_losses,
    'Validation MSE': validation_losses
})

# Save the DataFrame to a CSV file
loss_df.to_csv(f'{fd}/esm_model_train_validation_losses.csv', index=False)

#Save esm model
import pickle
with open(f'{fd}/esm_model.pkl', 'wb') as file:
    pickle.dump(esm_model, file)

# Plotting the training and validation loss
loss_df = pd.read_csv(f'{fd}/esm_model_train_validation_losses.csv')
plt.figure(figsize=(10, 6))
plt.plot(loss_df['Iteration'], loss_df['Training MSE'], label='Training Loss')
plt.plot(loss_df['Iteration'], loss_df['Validation MSE'], label='Validation Loss')
plt.xlabel('Iteration',fontsize=22, fontweight='bold')
plt.ylabel('Loss',fontsize=22, fontweight='bold')
plt.tick_params(axis='x', labelsize=18) # Changed to plt.tick_params and specified axis
plt.tick_params(axis='y', labelsize=18)
plt.legend()
plt.grid(True)
plt.savefig(f'{fd}/esm_model_train_validation_losses.png', dpi=300, bbox_inches='tight')
#plt.show()

pred_test_esm = esm_model.predict(weighted_test_preds.reshape(-1, 1))
pred_val_esm = esm_model.predict(weighted_val_preds.reshape(-1, 1))
pred_train_esm = esm_model.predict(weighted_train_preds.reshape(-1, 1))

# Calculate MAE
mae_test = mean_absolute_error(y_test, pred_test_esm)
mae_val = mean_absolute_error(y_val, pred_val_esm)
mae_train = mean_absolute_error(y_train, pred_train_esm)
print("Mean Absolute Error:", mae_test)
print("Mean Absolute Error:", mae_val)
print("Mean Absolute Error:", mae_train)

# R^2 (coefficient of determination) regression score function:
R2_test =r2_score(y_test, pred_test_esm)
R2_val =r2_score(y_val, pred_val_esm)
R2_train =r2_score(y_train, pred_train_esm)
print('R^2:', R2_test)
print('R^2:', R2_val)
print('R^2:', R2_train)

#MSE
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, pred_test_esm)
mse_val = mean_squared_error(y_val, pred_val_esm)
mse_train = mean_squared_error(y_train, pred_train_esm)
print('MSE:', mse_test)
print('MSE:', mse_val)
print('MSE:', mse_train)

#RMSE
rmse_test = mean_squared_error(y_test, pred_test_esm)
rmse_val = mean_squared_error(y_val, pred_val_esm)
rmse_train = mean_squared_error(y_train, pred_train_esm)

#Save the metrics in a csv file
import pandas as pd
df_test = pd.DataFrame({'MAE': [mae_test], 'R2': [R2_test], 'MSE': [mse_test], 'RMSE': [rmse_test]})
df_val = pd.DataFrame({'MAE': [mae_val], 'R2': [R2_val], 'MSE': [mse_val], 'RMSE': [rmse_val]})
df_train = pd.DataFrame({'MAE': [mae_train], 'R2': [R2_train], 'MSE': [mse_train], 'RMSE': [rmse_train]})

#Save df in csv file
df_test.to_csv(f'{fd}/esm_model_test_metrics.csv', index=False)
df_val.to_csv(f'{fd}/esm_model_val_metrics.csv', index=False)
df_train.to_csv(f'{fd}/esm_model_train_metrics.csv', index=False)

import pandas as pd

# Create DataFrame for actual vs predicted values for test set
df_test_predictions = pd.DataFrame({
    'Actual': y_test,  # Actual values
    'Predicted': pred_test_esm  # Predicted values
})

# Create DataFrame for actual vs predicted values for validation set
df_val_predictions = pd.DataFrame({
    'Actual': y_val,  # Actual values
    'Predicted': pred_val_esm  # Predicted values
})
df_train_predictions = pd.DataFrame({
    'Actual': y_train,  # Actual values
    'Predicted': pred_train_esm  # Predicted values
})

# Save the DataFrames to CSV
df_test_predictions.to_csv(f'{fd}/esm_test_actual_vs_predicted.csv', index=False)
df_val_predictions.to_csv(f'{fd}/esm_val_actual_vs_predicted.csv', index=False)
df_train_predictions.to_csv(f'{fd}/esm_train_actual_vs_predicted.csv', index=False)

print("CSV files saved successfully!")

#load the csv file
esm_plot=pd.read_csv(f'{fd}/esm_test_actual_vs_predicted.csv')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = esm_plot['Actual']
predicted = esm_plot['Predicted']

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)


fig, ax = plt.subplots(figsize=(10, 6))

hb1 = ax.hist2d(actual, predicted, bins=150, norm=colors.LogNorm(), cmap='plasma')
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
        color='red', linestyle='--', label='Ideal Prediction')
ax.set_xlabel('Actual Value', fontsize=18, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=18, fontweight='bold')

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

#ax.set_title('Actual vs. Predicted Values', fontsize=16
#ax.text(0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')

slope, intercept, r_value, _, _ = linregress(actual, predicted)
ax.text(0.05, 0.85, f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}', transform=ax.transAxes,
        fontsize=16, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


ax.text(0.60, 0.1, f'MAE = {mae:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.2, f'RMSE = {rmse:.4f} eV', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
ax.text(0.60, 0.3, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=16, verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cb = plt.colorbar(hb1[3], ax=ax, label='Count')
cb.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
cb.set_label('Count', fontsize=16, fontweight='bold')  # Adjust the font size of the colorbar label

plt.savefig(f'{fd}/esm_actual_vs_predicted_test.png', dpi=300, bbox_inches='tight')
#plt.show()
