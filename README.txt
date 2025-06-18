HOMO–LUMO Gap Prediction Framework
===================================

This repository provides scripts and models for predicting HOMO–LUMO gaps from molecular SMILES using machine learning. It includes descriptor-based feature extraction, model training, evaluation, SHAP-based interpretability, and HOMO-LUMO gap prediction.

Getting Started
---------------
1. Download the repository and place it in a suitable directory.
2. All required datasets are located in the `dataset/` directory.
3. All executable scripts are available in the `scripts/` directory.

Set Up the Conda Environment
----------------------------
    conda env create -f homolumo_env.yml
    conda activate homolumo_env

Training and Prediction
-----------------------
Training Command:

    python scripts/training.py

- All outputs will be saved in the `Output/` directory.
- To use a different random seed, edit the `random_state` of each model in `scripts/training_and_prediction.py`.

Notes:
- A subset of the GDB13 dataset is used. It is located at: `dataset/GDB13_subset/`
- The dataset is already split into training, validation, and test sets.
- Feature selection includes RDKit descriptor calculation and correlation filtering.
- Evaluation metrics include: MAE, MSE, RMSE, and R².
- Early stopping is applied during training to prevent overfitting.
- Ensemble predictions are a weighted average of:
    - LGBM (0.7)
    - Bi-LSTM (0.2)
    - MLP (0.1)

SHAP Analysis (Model Interpretability)
--------------------------------------
Command:

    python scripts/SHAP_analysis.py

This performs SHAP analysis on trained CatBoost and LightGBM models and generates both summary and instance-level visualizations.

HOMO–LUMO Gap Predictions
-------------------------
Input File Format:
- Input must be a CSV file with at least one column of SMILES strings.
- The default column name is `SMILES`, but you can change it with the `-c` flag.

1. Predict HF + EXP Gaps (Applies empirical conversion)

    python scripts/predict_gap_from_file.py -i input.csv -o results.csv

    With a custom SMILES column:

    python scripts/predict_gap_from_file.py -i input.csv -o results.csv -c smiles_input

2. Predict HF Gap Only (No empirical conversion)

    python scripts/predict_hf_only.py -i input.csv -o hf_only.csv

Important Notes
---------------
- Ensure all required model files are present in the `ML_weights/` directory before running any prediction or SHAP scripts.
- Training example output files are kept in Output/examples and SHAP analysis example output can be found in Output/SHAP/examples directory.
