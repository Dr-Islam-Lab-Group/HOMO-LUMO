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

# Fixed location for trained model folder
MODEL_DIR = "ML_weights"

imp_features = (
    'FractionCSP3', 'SMR_VSA7', 'BCUT2D_MRHI', 'HallKierAlpha',
    'MinAbsEStateIndex', 'fr_allylic_oxid', 'SMR_VSA10', 'MaxAbsEStateIndex',
    'PEOE_VSA5', 'SMR_VSA9', 'MaxPartialCharge', 'BCUT2D_MWHI',
    'NumAmideBonds', 'BCUT2D_MWLOW', 'SMR_VSA3', 'MolMR', 'PEOE_VSA7',
    'MinEStateIndex', 'BCUT2D_CHGLO', 'fr_NH0'
)

# Load models and scaler
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

HALOGENS = {"F", "Cl", "Br", "I"}

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

def count_atoms_and_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0, 0, 0]
    counts = {"C": 0, "N": 0, "O": 0, "S": 0, "X": 0}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        elif sym in HALOGENS:
            counts["X"] += 1
    pi_b = sum([
        0.5 if bond.GetIsAromatic() else max(0, int(round(bond.GetBondTypeAsDouble())) - 1)
        for bond in mol.GetBonds()
    ])
    return [counts[k] for k in ("C", "N", "O", "S", "X")] + [pi_b]

# SMARTS substructure matching
sma_acetylene = Chem.MolFromSmarts("[*]C#C[*]")
sma_polyene   = Chem.MolFromSmarts("C=C-C=C")
sma_thiophene = Chem.MolFromSmarts("c1ccsc1")
sma_pyrrole   = Chem.MolFromSmarts("[nH]1cccc1")

def belongs_to_eq1(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any([
        mol.HasSubstructMatch(sma_acetylene),
        mol.HasSubstructMatch(sma_polyene),
        mol.HasSubstructMatch(sma_thiophene),
        mol.HasSubstructMatch(sma_pyrrole)
    ])

def predict_gaps(smiles_list):
    desc_df = calculate_descriptors(smiles_list)
    X_scaled = scaler.transform(desc_df)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    lgbm_p = lgbm_model.predict(X_scaled)
    lstm_p = Bilstm_model.predict(X_reshaped).flatten()
    mlp_p = mlp_model.predict(X_scaled)
    hf_pred = 0.7 * lgbm_p + 0.2 * lstm_p + 0.1 * mlp_p
    hf_meta = esm_model.predict(hf_pred.reshape(-1, 1)).flatten()

    results = []
    for smi, hf_val in zip(smiles_list, hf_meta):
        hf_val = round(float(hf_val), 4)
        c, n, o, s, x, pi_b = count_atoms_and_bonds(smi)
        if belongs_to_eq1(smi):
            exp_val = 0.0 if c == 0 else 2.1414 + (13.6365 / c) + 0.0154 * hf_val
        else:
            exp_val = 3.8492 - 0.0548 * c - 0.1004 * n - 0.0612 * o - 0.0709 * x + 0.0314 * pi_b + 0.0372 * hf_val
        exp_val = round(exp_val, 4)
        results.append((smi, hf_val, exp_val))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict HOMO–LUMO Gaps from SMILES in a CSV file.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file with a column of SMILES strings")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file to write predictions")
    parser.add_argument("-c", "--column", default="SMILES", help="Column name in input file (default: SMILES)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in input file.")

    smiles_list = df[args.column].astype(str).tolist()
    predictions = predict_gaps(smiles_list)

    df_out = pd.DataFrame(predictions, columns=["SMILES", "HF_Pred_eV", "EXP_Pred_eV"])
    df_out.to_csv(args.output, index=False)
    print(f"[✓] Predictions saved to: {args.output}")

