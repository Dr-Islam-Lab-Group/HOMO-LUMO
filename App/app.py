# app.py
# -*- coding: utf-8 -*-
"""Web service for predicting HOMO–LUMO gaps with an ensemble model.
This version (May 2025) uses an updated π‑bond counting routine:
    π‑B = 1 × double + 2 × triple + 3 × quadruple + 0.5 × aromatic bonds
and counts C, N, O, S and halogens (grouped as X).
Only C, N, O, X and π‑B feed into eq (2); S is returned for completeness.
"""

import warnings
warnings.filterwarnings("ignore")

###############################################
# 1) Imports
###############################################
from flask import Flask, request, jsonify, render_template_string
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

###############################################
# 2) Flask Setup
###############################################
app = Flask(__name__)

###############################################
# 3) Important Feature List (20 descriptors)
###############################################
imp_features = (
    'FractionCSP3', 'SMR_VSA7', 'BCUT2D_MRHI', 'HallKierAlpha',
    'MinAbsEStateIndex', 'fr_allylic_oxid', 'SMR_VSA10', 'MaxAbsEStateIndex',
    'PEOE_VSA5', 'SMR_VSA9', 'MaxPartialCharge', 'BCUT2D_MWHI',
    'NumAmideBonds', 'BCUT2D_MWLOW', 'SMR_VSA3', 'MolMR', 'PEOE_VSA7',
    'MinEStateIndex', 'BCUT2D_CHGLO', 'fr_NH0'
)

###############################################
# 4) Load models & scaler
###############################################
with open('LGBM_model_weights.txt', 'rb') as f:
    lgbm_model = pickle.load(f)
with open('mlp1_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)
with open('esm_model.pkl', 'rb') as f:
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
Bilstm_model.load_weights('Bi_LSTM_model.weights.h5')

with open('my_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

###############################################
# 5) Helper functions
###############################################
HALOGENS = {"F", "Cl", "Br", "I"}

def calculate_descriptors(smiles_list):
    """Return a DataFrame with the 20 selected descriptors for each SMILES."""
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
    """Return [C, N, O, S, X, pi_B] following the new π‑bond definition."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0, 0, 0]  # graceful fallback

    counts = {"C": 0, "N": 0, "O": 0, "S": 0, "X": 0}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
        elif sym in HALOGENS:
            counts["X"] += 1

    pi_b = 0.0
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            pi_b += 0.5
        else:
            order = int(round(bond.GetBondTypeAsDouble()))  # 1,2,3,4
            if order > 1:
                pi_b += (order - 1)  # double→1, triple→2, quadruple→3
    return [counts[k] for k in ("C", "N", "O", "S", "X")] + [pi_b]

# Substructure SMARTS for eq (1) classification
sma_acetylene = Chem.MolFromSmarts("[*]C#C[*]")
sma_polyene   = Chem.MolFromSmarts("C=C-C=C")
sma_thiophene = Chem.MolFromSmarts("c1ccsc1")
sma_pyrrole   = Chem.MolFromSmarts("[nH]1cccc1")

def belongs_to_eq1(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any([
        mol.HasSubstructMatch(sma_acetylene),
        mol.HasSubstructMatch(sma_polyene),
        mol.HasSubstructMatch(sma_thiophene),
        mol.HasSubstructMatch(sma_pyrrole)
    ])

###############################################
# 6) HTML template (unchanged)
###############################################
html_template = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\" />
<title>HOMO–LUMO Prediction</title>
<style>
 body{background:#f7f7f7;font-family:Arial,sans-serif;margin:0}
 .container{width:80%;max-width:800px;margin:40px auto;background:#fff;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,.1);padding:20px 30px}
 h1{text-align:center;color:#444}
 .input-section{margin:20px 0;text-align:center}
 .input-section label{display:block;font-weight:600;margin-bottom:8px;color:#555}
 .input-section input{width:80%;max-width:600px;padding:10px;font-size:16px;border:1px solid #ccc;border-radius:6px;margin-bottom:15px}
 .input-section button{padding:10px 20px;font-size:16px;background:#4285f4;color:#fff;border:none;border-radius:6px;cursor:pointer}
 .input-section button:hover{background:#306ac2}
 table{width:100%;border-collapse:collapse;margin-top:25px}
 thead{background:#4285f4;color:#fff}
 th,td{padding:12px 8px;border:1px solid #ddd;text-align:center}
 tr:nth-child(even){background:#f9f9f9}
 .footer{text-align:center;margin-top:40px;font-size:14px;color:#888}
</style>
</head>
<body>
<div class=\"container\">
 <h1>Predict HOMO–LUMO Gap</h1>
 <div class=\"input-section\">
  <form action=\"/predict\" method=\"post\">
   <label for=\"smiles\">Enter SMILES (comma‑separated):</label>
   <input id=\"smiles\" name=\"smiles\" required />
   <br/>
   <button type=\"submit\">Predict</button>
  </form>
 </div>
 {% if predictions %}
 <table>
  <thead><tr><th>SMILES</th><th>Pred. HF Gap&nbsp;(eV)</th><th>Pred. Exp. Gap&nbsp;(eV)</th></tr></thead>
  <tbody>
  {% for item in predictions %}
   <tr><td>{{item.SMILES}}</td><td>{{item.HF}}</td><td>{{item.EXP}}</td></tr>
  {% endfor %}
  </tbody>
 </table>
 {% endif %}
</div>
<div class=\"footer\">&copy; Dr. M. Shahidul Islam Research Lab, DSU (Dover, DE)</div>
</body></html>
"""

###############################################
# 7) Flask routes
###############################################
@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        smiles_input = request.form['smiles']
        smiles_list = [s.strip() for s in smiles_input.split(',') if s.strip()]
        if not smiles_list:
            return jsonify({'error': 'No SMILES supplied.'}), 400

        # Descriptor matrix for ML models
        desc_df = calculate_descriptors(smiles_list)
        X_scaled = scaler.transform(desc_df)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Ensemble HF prediction
        lgbm_p = lgbm_model.predict(X_scaled)
        lstm_p = Bilstm_model.predict(X_reshaped).flatten()
        mlp_p  = mlp_model.predict(X_scaled)
        hf_pred = 0.7*lgbm_p + 0.2*lstm_p + 0.1*mlp_p

        # Gradient‑boost meta‑model
        hf_meta = esm_model.predict(hf_pred.reshape(-1,1)).flatten()

        results = []
        for smi, hf_val in zip(smiles_list, hf_meta):
            hf_val = round(float(hf_val), 4)
            c,n,o,s,x,pi_b = count_atoms_and_bonds(smi)

            if belongs_to_eq1(smi):
                exp_val = 0.0 if c==0 else 2.1414 + (13.6365/c) + 0.0154*hf_val
            else:
                exp_val = 3.8492 - 0.0548*c - 0.1004*n - 0.0612*o - 0.0709*x + 0.0314*pi_b + 0.0372*hf_val
            exp_val = round(exp_val, 4)
            results.append(dict(SMILES=smi, HF=hf_val, EXP=exp_val))

        return render_template_string(html_template, predictions=results)

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

###############################################
###############################################
# 9) Run the Flask Application
###############################################
import os

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

if __name__ == "__main__":
    run_flask()  # Run in main thread (not in background)
