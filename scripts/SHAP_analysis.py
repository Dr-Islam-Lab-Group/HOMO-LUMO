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


from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

# Output Directory
fd = "Output/SHAP"

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

#import pandas as pd
import matplotlib.pyplot as plt
#from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Initialize the CatBoostRegressor with the best hyperparameters
final_model_CatB = CatBoostRegressor(
    iterations=4000,
    depth=10,
    learning_rate=0.08003866739082473,
    l2_leaf_reg=1.2219300331335978,
    bagging_temperature=0.9026337873010564,
    random_seed=42,
    verbose=100,  # Log progress every 100 iterations
    eval_metric='MAE',  # Use MAE as evaluation metric
    loss_function='RMSE'  # Use RMSE as the loss function
)

# save the trained model weight
#final_model_CatB.save_model(f'{fd}/CatBoost_model.weights.h5')

#load saved model
final_model_CatB.load_model('ML_weights/CatBoost_model.weights.h5')

import shap
import matplotlib.pyplot as plt

# Initialize SHAP
shap.initjs()

# Compute SHAP values
explainer = shap.TreeExplainer(final_model_CatB)
shap_values = explainer.shap_values(X_test)

# Define range for analysis
start_index = 0
end_index = 1000
local_shap_values = shap_values[start_index:end_index]

# Make predictions
prediction = final_model_CatB.predict(X_test[start_index:end_index])
print(f"The final_model_CatB Regressor predicted: {prediction}")

# Force plot (requires JS visualization support)
#shap.force_plot(explainer.expected_value, local_shap_values, X_test[start_index:end_index])

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)

# Summary plot (save properly)
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)
plt.savefig(f'{fd}/catB_model_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close to avoid overlapping plots


# Individual force plot for first instance
shap.force_plot(explainer.expected_value, local_shap_values[0], X_test.iloc[0], feature_names=X_train.columns)

# Ensure SHAP JS visualizations are initialized
shap.initjs()

# Define SHAP Explanation object
shap_expl = shap.Explanation(
    values=local_shap_values[0],  # SHAP values for the specific instance
    base_values=explainer.expected_value,  # Base value
    data=X_test.iloc[0]  # Feature values for the instance
)

# Create a new figure before plotting
plt.figure(figsize=(8, 6))  # Set figure size
shap.waterfall_plot(shap_expl)  # Generate waterfall plot

# Save the waterfall plot
plt.savefig(f'{fd}/catB_waterfall_plot_instance_0.png', dpi=300, bbox_inches='tight')
plt.close()  # Close figure to avoid overlapping plots

print("SHAP Waterfall plot saved successfully as 'shap_waterfall_plot_instance_0.png'.")


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
    random_state=777,
    verbose=100,
    objective='regression',  # Default for regression (MSE loss)
)



#load saved model
import pickle
filename = 'ML_weights/LGBM_model_weights.txt'
lgbm_model = pickle.load(open(filename, 'rb'))

import shap
import matplotlib.pyplot as plt

# Initialize SHAP
shap.initjs()

# Compute SHAP values for LGBM model
explainer_lgbm = shap.TreeExplainer(lgbm_model)
shap_values_lgbm = explainer_lgbm.shap_values(X_test)

# Define range for analysis
start_index = 0
end_index = 1000
local_shap_values_lgbm = shap_values_lgbm[start_index:end_index]

# Make predictions
prediction_lgbm = lgbm_model.predict(X_test[start_index:end_index])
print(f"The LGBM Regressor predicted: {prediction_lgbm}")

# Summary plot (save properly)
shap.summary_plot(shap_values_lgbm, X_test, feature_names=X_train.columns)
plt.savefig(f'{fd}/lgbm_model_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close to avoid overlapping plots

# Generate and save the SHAP Waterfall plot
shap_expl = shap.Explanation(
    values=local_shap_values_lgbm[0],  # SHAP values for a single instance
    base_values=explainer_lgbm.expected_value,  # Base value
    data=X_test.iloc[0]  # Feature values for the instance
)

# Create a new figure before plotting
plt.figure(figsize=(8, 6))
shap.waterfall_plot(shap_expl)
plt.savefig(f'{fd}/lgbm_model_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to prevent issues

print("SHAP waterfall plot saved successfully as 'lgbm_model_waterfall_plot.png'.")

