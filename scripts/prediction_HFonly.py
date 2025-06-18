# predict_hf_only.py

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

MODEL_DIR = "ML_weights"

imp_features = (
    'FractionCSP3', 'SMR_VSA7', 'BCUT2D_MRHI', 'HallKierAlpha',
    'MinAbsEStateIndex', 'fr_allylic_oxid', 'SMR_VSA10', 'MaxAbsEStateIndex',
    'PEOE_VSA5', 'SMR_VSA9', 'MaxPartialCharge', 'BCUT2D_MWHI',
    'NumAmideBonds', 'BCUT2D_MWLOW', 'SMR_VSA3', 'MolMR', 'PEOE_VSA7',
    'MinEStateIndex', 'BCUT2D_CHGLO', 'fr_NH0'
)

with open(os.path.join(MODEL_DIR, 'LGBM_model_weights.txt'), 'rb') as f:
    lgbm_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'mlp1_model.pkl'), 'rb') as f:
    mlp_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'esm_model.pkl'), 'rb') as f:
    esm_model = pickle.load(f)

num_features = len(imp_features)
Bilstm_model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(1, num_features)),
    Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
    Bidirectional(LSTM(64, activation='relu')),
    Dense(1)
])
Bilstm_model.compile(optimizer='adam', loss='mean_squared_error')
Bilstm_model.load_weights(os.path.join(MODEL_DIR, 'Bi_LSTM_model.weights.h5'))

with open(os.path.join(MODEL_DIR, 'my_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

def calculate_descriptors(smiles_list):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([d[0] for d in Descriptors._descList])
    names = calc.GetDescriptorNames()

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = Chem.AddHs(mol)
            rows.append(calc.CalcDescriptors(mol))
        else:
            rows.append([None] * len(names))

    df = pd.DataFrame(rows, columns=names)
    for col in imp_features:
        if col not in df.columns:
            df[col] = 0
    df = df[list(imp_features)].fillna(0).replace(-np.inf, 0)
    return df

def predict_hf(smiles_list):
    desc_df = calculate_descriptors(smiles_list)
    X_scaled = scaler.transform(desc_df)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    lgbm_p = lgbm_model.predict(X_scaled)
    lstm_p = Bilstm_model.predict(X_reshaped).flatten()
    mlp_p = mlp_model.predict(X_scaled)
    hf_pred = 0.7 * lgbm_p + 0.2 * lstm_p + 0.1 * mlp_p
    hf_meta = esm_model.predict(hf_pred.reshape(-1, 1)).flatten()

    return [(smi, round(float(hf), 4)) for smi, hf in zip(smiles_list, hf_meta)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict HF HOMO–LUMO Gaps from SMILES in a CSV file.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file with a column of SMILES strings")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file to write predictions")
    parser.add_argument("-c", "--column", default="SMILES", help="Column name in input file (default: SMILES)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in input file.")

    smiles_list = df[args.column].astype(str).tolist()
    predictions = predict_hf(smiles_list)

    df_out = pd.DataFrame(predictions, columns=["SMILES", "HF_Pred_eV"])
    df_out.to_csv(args.output, index=False)
    print(f"[✓] HF predictions saved to: {args.output}")

