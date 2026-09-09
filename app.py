"""
Quantum Error Prediction — Streamlit App
------------------------------------------
Loads the trained XGBoost multiclass classifier
(`quantum_error_prediction_ml_model`) and predicts which type of error
a quantum circuit execution is likely to exhibit, from circuit / gate /
hardware telemetry features.

Run:
    streamlit run app.py

Required file in the same folder:
    quantum_error_prediction_ml_model   (joblib-saved XGBClassifier)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Error Prediction", page_icon="⚛️", layout="wide")

# ------------------------------------------------------------------ #
# Constants reconstructed from the training notebook
# (target classes are alphabetically ordered — this is what
# sklearn's LabelEncoder produced when the model was trained)
# ------------------------------------------------------------------ #
ERROR_LABELS = ["amplitude_damping", "depolarizing", "leakage", "none", "phase_damping", "readout"]
GATE_TYPES = ["CZ", "H", "Z", "X", "RZ", "CNOT", "RY", "SX", "Y", "SWAP", "RX", "CCX"]

# Exact column order the model was trained on. NOTE: the gate_* one-hot
# columns are alphabetical (pandas get_dummies sorts categories), which is
# NOT the same order as GATE_TYPES above (that list is just UI display order
# taken from the notebook's df['gate_type'].unique() — do not derive the
# training column order from it, they differ and the model is order-sensitive).
FEATURE_ORDER = [
    "qubit_count", "gate_depth", "cnot_gate_count", "single_qubit_gate_count", "swap_gate_count",
    "measurement_count", "gate_fidelity", "readout_fidelity", "error_rate_gate", "readout_error",
    "t1_time_us", "t2_time_us", "gate_duration_ns", "execution_time_us", "idle_time_us",
    "calibration_age_hr", "qubit_connectivity", "crosstalk_level", "temperature_mK", "shots",
    "bitstring", "ideal_bitstring", "fidelity", "depth_per_qubit", "t2_t1_ratio",
    "gate_density", "noise_score",
] + [f"gate_{g}" for g in sorted(GATE_TYPES)]


@st.cache_resource
def load_model():
    return joblib.load("quantum error prediction ml model")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model = load_model()
explainer = load_explainer(model)


def build_feature_row(values: dict, gate_type: str) -> pd.DataFrame:
    row = dict(values)
    for g in GATE_TYPES:
        row[f"gate_{g}"] = 1 if g == gate_type else 0
    return pd.DataFrame([row])[FEATURE_ORDER]


# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #
st.title("⚛️ Quantum Error Prediction")
st.caption("XGBoost multiclass model · predicts the likely `error_type` for a quantum circuit run "
           "from gate, fidelity, timing and readout telemetry")

tab_single, tab_batch, tab_about = st.tabs(["🔮 Single Prediction", "📄 Batch Prediction", "ℹ️ About"])

with tab_single:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Circuit structure")
        qubit_count = st.slider("qubit_count", 2, 127, 65)
        gate_depth = st.slider("gate_depth", 20, 800, 410)
        cnot_gate_count = st.number_input("cnot_gate_count", 0, 400, 92)
        single_qubit_gate_count = st.number_input("single_qubit_gate_count", 0, 900, 318)
        swap_gate_count = st.number_input("swap_gate_count", 0, 20, 7)
        measurement_count = st.slider("measurement_count", 2, 127, qubit_count,
                                       help="Typically equals qubit_count in the training data")
        gate_type = st.selectbox("gate_type", GATE_TYPES, index=0)
        qubit_connectivity = st.number_input("qubit_connectivity", 0, 127, 4)

    with col2:
        st.subheader("Fidelity & noise")
        gate_fidelity = st.slider("gate_fidelity", 0.980, 1.000, 0.993, format="%.4f")
        readout_fidelity = st.slider("readout_fidelity", 0.940, 1.000, 0.972, format="%.4f")
        error_rate_gate = st.slider("error_rate_gate", 0.0000, 0.0200, 0.0070, format="%.4f")
        readout_error = st.slider("readout_error", 0.0000, 0.0600, 0.0276, format="%.4f")
        crosstalk_level = st.slider("crosstalk_level", 0.0000, 0.1000, 0.0500, format="%.4f")
        fidelity = st.slider("fidelity", 0.50, 0.95, 0.736, format="%.3f")
        t2_t1_ratio = st.slider("t2_t1_ratio", 0.28, 0.95, 0.78, format="%.3f")
        gate_density = st.slider("gate_density", 0.00, 0.45, 0.223, format="%.3f")
        noise_score = st.slider("noise_score", 0.00, 1.00, 0.93, format="%.3f")
        depth_per_qubit = st.number_input("depth_per_qubit", 0.0, 400.0, 6.35, format="%.3f")

    with col3:
        st.subheader("Hardware / timing")
        t1_time_us = st.number_input("t1_time_us", 0.0, 500.0, 125.0)
        t2_time_us = st.number_input("t2_time_us", 0.0, 500.0, 55.3)
        gate_duration_ns = st.number_input("gate_duration_ns", 0.0, 2000.0, 235.0)
        execution_time_us = st.number_input("execution_time_us", 0.0, 20000.0, 3890.0)
        idle_time_us = st.number_input("idle_time_us", 0.0, 10000.0, 1400.0)
        calibration_age_hr = st.number_input("calibration_age_hr", 0.0, 200.0, 19.5)
        temperature_mK = st.slider("temperature_mK", 10.0, 18.0, 14.0, format="%.2f")
        shots = st.selectbox("shots", [1024, 2048, 4096, 8192], index=1)
        bitstring = st.number_input("bitstring (raw, as trained)", value=1111111, step=1,
                                     help="Trained on bit patterns parsed as plain integers "
                                          "(leading zeros not preserved) — enter in the same form.")
        ideal_bitstring = st.number_input("ideal_bitstring (raw, as trained)", value=1111111, step=1)

    st.divider()
    if st.button("Predict Error Type", type="primary", use_container_width=True):
        values = dict(
            qubit_count=qubit_count, gate_depth=gate_depth, cnot_gate_count=cnot_gate_count,
            single_qubit_gate_count=single_qubit_gate_count, swap_gate_count=swap_gate_count,
            measurement_count=measurement_count, gate_fidelity=gate_fidelity,
            readout_fidelity=readout_fidelity, error_rate_gate=error_rate_gate,
            readout_error=readout_error, t1_time_us=t1_time_us, t2_time_us=t2_time_us,
            gate_duration_ns=gate_duration_ns, execution_time_us=execution_time_us,
            idle_time_us=idle_time_us, calibration_age_hr=calibration_age_hr,
            qubit_connectivity=qubit_connectivity, crosstalk_level=crosstalk_level,
            temperature_mK=temperature_mK, shots=shots, bitstring=bitstring,
            ideal_bitstring=ideal_bitstring, fidelity=fidelity, depth_per_qubit=depth_per_qubit,
            t2_t1_ratio=t2_t1_ratio, gate_density=gate_density, noise_score=noise_score,
        )
        X = build_feature_row(values, gate_type)

        pred_class = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        st.success(f"### Predicted error type: **{ERROR_LABELS[pred_class]}**")

        proba_df = pd.DataFrame({"error_type": ERROR_LABELS, "probability": proba}).sort_values(
            "probability", ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Class probabilities")
            st.bar_chart(proba_df.set_index("error_type"))
        with c2:
            st.subheader("Why this prediction? (SHAP)")
            shap_values = explainer.shap_values(X)
            # Newer SHAP: array shape (1, n_features, n_classes). Older: list per class.
            if isinstance(shap_values, list):
                class_shap = shap_values[pred_class][0]
                base_val = np.atleast_1d(explainer.expected_value)[pred_class]
            elif shap_values.ndim == 3:
                class_shap = shap_values[0, :, pred_class]
                base_val = np.atleast_1d(explainer.expected_value)[pred_class]
            else:
                class_shap = shap_values[0]
                base_val = np.atleast_1d(explainer.expected_value)[0]

            fig, ax = plt.subplots(figsize=(7, 5))
            shap.waterfall_plot(
                shap.Explanation(values=class_shap, base_values=base_val,
                                  data=X.iloc[0], feature_names=X.columns.tolist()),
                show=False, max_display=10,
            )
            st.pyplot(fig, use_container_width=True)

# ---------------- Batch prediction tab ---------------- #
with tab_batch:
    st.write(f"Upload a CSV with these {len(FEATURE_ORDER) - len(GATE_TYPES) + 1} columns: "
             f"the numeric features below, plus a single `gate_type` column "
             f"(one of {', '.join(GATE_TYPES)}) instead of the twelve `gate_*` one-hot columns "
             "— the app builds the one-hot encoding for you.")
    st.code(", ".join(FEATURE_ORDER[: -len(GATE_TYPES)] + ["gate_type"]))

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = set(FEATURE_ORDER[: -len(GATE_TYPES)] + ["gate_type"]) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
        else:
            rows = []
            for _, r in batch_df.iterrows():
                values = {c: r[c] for c in FEATURE_ORDER[: -len(GATE_TYPES)]}
                rows.append(build_feature_row(values, r["gate_type"]))
            X_all = pd.concat(rows, ignore_index=True)
            preds = model.predict(X_all)
            out = batch_df.copy()
            out["Predicted_error_type"] = [ERROR_LABELS[p] for p in preds]
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions as CSV", out.to_csv(index=False),
                                file_name="predictions.csv", mime="text/csv")

# ---------------- About tab ---------------- #
with tab_about:
    st.markdown(f"""
    **Model:** XGBoost multiclass classifier (`scale_pos_weight`-adjusted, 100 trees, depth 4)
    trained to predict `error_type` for a quantum circuit execution.

    **Classes:** {', '.join(ERROR_LABELS)}
    (alphabetical order — this is the order scikit-learn's `LabelEncoder` produced during training,
    and it's the order the model's raw class indices correspond to.)

    **Pipeline notes**
    - `circuit_id`, `device_family`, and `device_version` were dropped before training and are not
      needed here.
    - `gate_type` was one-hot encoded from a single categorical column — this app reproduces that
      by setting exactly one `gate_*` column to 1 based on your selection.
    - `bitstring` / `ideal_bitstring` were parsed as plain integers in training (binary-looking
      strings like `"01110"` lose their leading zero this way) — enter values in that same raw
      integer form for consistent predictions.
    - Ranges/defaults for the sliders above come from the training data's summary statistics;
      a few timing fields (T1/T2, gate duration, execution/idle time) didn't have their
      min/max captured in the source notebook, so those use plausible defaults — feel free to
      adjust them for your own hardware relative to the fixed presets.
    """)
