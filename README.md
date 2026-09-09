# # ⚛️ Quantum Error Prediction & Classification

Predicts the type of error a quantum circuit execution is likely to exhibit — from gate, fidelity, timing, and readout telemetry — using an XGBoost multiclass classifier served through an interactive Streamlit app.

## Overview

Quantum hardware runs are noisy: gate errors, decoherence, crosstalk, and readout mistakes all show up as different failure signatures. This project trains a multiclass model to classify a circuit run into one of six outcomes and explains *why* using SHAP:

- **Error classes:** `amplitude_damping`, `depolarizing`, `leakage`, `none`, `phase_damping`, `readout`
- **Single prediction** — enter circuit structure, fidelity/noise, and hardware/timing parameters to get a predicted error type with class probabilities.
- **Batch prediction** — upload a CSV of circuit runs (with a plain `gate_type` column) and get predictions for the whole batch.
- **SHAP explanations** — waterfall plot showing which telemetry features drove the predicted class.
- **Tableau dashboard** — a companion `.twb` workbook for exploring error patterns visually.

## Repository contents

| File | Purpose |
|---|---|
| `quantam project v4.ipynb` | EDA, feature engineering, model training notebook |
| `app.py` | Streamlit app — single & batch prediction, SHAP explainability |
| `quantum error prediction ml model` | Trained XGBoost classifier, saved with `joblib` |
| `quantum error prediction and classification.twb` | Tableau dashboard workbook |
| `tableau quantum dashboard.PNG` | Dashboard preview image |
| `jupyter.PNG` | Notebook preview image |
| `QuantumPredict [Autosaved].pptx` | Project summary slides |

## Model details

- **Algorithm:** XGBoost multiclass classifier, `scale_pos_weight`-adjusted, 100 trees, max depth 4
- **Features:** 27 numeric circuit/hardware telemetry fields (qubit count, gate depth, gate/readout fidelity, T1/T2 times, crosstalk, noise score, etc.) plus a one-hot encoded `gate_type`
- **Class order:** alphabetical, matching scikit-learn's `LabelEncoder` from training
- Dropped columns not used for prediction: `circuit_id`, `device_family`, `device_version`

## Getting started

```bash
pip install streamlit joblib shap matplotlib pandas numpy xgboost
streamlit run app.py
```

Make sure `quantum error prediction ml model` is in the same folder as `app.py`. For batch prediction, your CSV needs the numeric feature columns plus a single `gate_type` column (one of `CZ, H, Z, X, RZ, CNOT, RY, SX, Y, SWAP, RX, CCX`) — the app builds the one-hot encoding for you.

## Tech stack

Python · pandas · XGBoost (multiclass) · SHAP · Streamlit · Tableau

## Author

Divakaran A P
