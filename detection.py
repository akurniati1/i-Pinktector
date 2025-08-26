import streamlit as st
import numpy as np
import tensorflow as tf
import time
import zipfile
import os
import pdfplumber
import re
from functools import lru_cache


#Replace 'file.zip' with the name of your ZIP file
zip_file_path = 'C-Tr-SVD.zip'
extract_to_directory = 'C-Tr-SVD'

#Extract the ZIP file
if not os.path.exists(extract_to_directory):
    os.makedirs(extract_to_directory)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_directory)

#Load the trained model using TFSMLayer
class SVDLayer2(tf.keras.layers.Layer):
    def __init__(self, weight, type_svd, rank, activation=None, **kwargs):
        super(SVDLayer2, self).__init__(**kwargs)
        self.U, self.S, self.V = self.svd_decomposition(weight, type_svd, rank)
        self.activation = tf.keras.activations.get(activation)

    def svd_decomposition(self, weight, type_svd, rank):
        Ur = np.random.rand(weight.shape[0], rank)
        Sr = np.random.rand(rank, rank)
        Vr = np.random.rand(rank, weight.shape[1])
        return Ur, Sr, Vr

    def build(self, input_shape):
        self.U = tf.Variable(initial_value=tf.constant(self.U), trainable=True)
        self.S = tf.Variable(initial_value=tf.constant(self.S), trainable=True)
        self.V = tf.Variable(initial_value=tf.constant(self.V), trainable=True)

    def call(self, inputs):
        result = tf.matmul(inputs, self.U)
        result = tf.matmul(result, self.S)
        result = tf.matmul(result, self.V)
        if self.activation is not None:
            result = self.activation(result)
        return result

    def get_config(self):
        config = super(SVDLayer2, self).get_config()
        config.update({
            'U': self.U.numpy(),
            'S': self.S.numpy(),
            'V': self.V.numpy(),
            'activation': tf.keras.activations.serialize(self.activation),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['U'] = tf.constant(config.pop('U'))
        config['S'] = tf.constant(config.pop('S'))
        config['V'] = tf.constant(config.pop('V'))
        config['activation'] = tf.keras.activations.deserialize(config['activation'])
        return cls(**config)

model = tf.keras.layers.TFSMLayer(extract_to_directory, call_endpoint='serving_default')

#Predict function
def predict(input_features):
    prediction = model(input_features)
    return prediction

#Decimal formatting
DECIMALS = 4
def fmt_num_dot(x: float) -> str:
    """Format to 4 decimals using dot as decimal separator."""
    return f"{x:.{DECIMALS}f}"

#Fields & Labels
meta_labels = {"patient_id": "Patient ID", "date": "Date"}

base_params = [
    "radius", "texture", "perimeter", "area",
    "smoothness", "compactness", "concavity",
    "concave_points", "symmetry", "fractal_dimension"
]

features_mean_params  = [f"{p}_mean"  for p in base_params]
features_se_params    = [f"{p}_se"    for p in base_params]
features_worst_params = [f"{p}_worst" for p in base_params]

labels_mean_params  = {f"{p}_mean":  f"{p.replace('_',' ').title()} Mean"  for p in base_params}
labels_se_params    = {f"{p}_se":    f"{p.replace('_',' ').title()} SE"    for p in base_params}
labels_worst_params = {f"{p}_worst": f"{p.replace('_',' ').title()} Worst" for p in base_params}

#Regex builders
@lru_cache(maxsize=None)
def _label_pattern(label: str) -> re.Pattern:
    """Toleran spasi/TAB/NBSP di label & sekitar separator, menangkap angka dot/comma."""
    parts = re.split(r"\s+", label.strip())
    flexible_label = r"[\s\u00A0\t]+".join(map(re.escape, parts))
    number = r"([-+]?\d+(?:[.,]\d+)?)"
    return re.compile(
        rf"{flexible_label}[\s\u00A0\t]*[:：=][\s\u00A0\t]*{number}",
        flags=re.IGNORECASE
    )

@lru_cache(maxsize=None)
def _meta_pattern(label: str) -> re.Pattern:
    parts = re.split(r"\s+", label.strip())
    flexible_label = r"[\s\u00A0\t]+".join(map(re.escape, parts))
    return re.compile(
        rf"{flexible_label}[\s\u00A0\t]*[:：=][\s\u00A0\t]*([^\n\r]+)",
        flags=re.IGNORECASE
    )

def _parse_number(s: str):
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        return None

#PDF parsing
ALL_LABELS = {
    **labels_mean_params,
    **labels_se_params,
    **labels_worst_params,
}

def parse_pdf(file):
    vals, meta = {}, {}

    with pdfplumber.open(file) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    # normalisasi NBSP → spasi biasa
    text = text.replace("\u00A0", " ")

    # fitur (mean, se, worst) dalam satu loop
    for key, label in ALL_LABELS.items():
        m = _label_pattern(label).search(text)
        if not m:
            continue
        num = _parse_number(m.group(1))
        vals[key] = None if num is None else round(num, DECIMALS)

    # metadata
    for mkey, mlabel in meta_labels.items():
        mm = _meta_pattern(mlabel).search(text)
        if mm:
            meta[mkey] = mm.group(1).strip()

    return vals, meta

#Set for read only fields
def locked_text(label, value_key, placeholder):
    """Read-only text input for meta/string values, shows placeholder when empty."""
    st.text_input(
        label,
        value=st.session_state.get(value_key, "") or "",
        key=f"tx_{value_key}",
        disabled=True,
        placeholder=placeholder,
    )

def locked_text_numeric(label, value_key):
    """
    Read-only text input for numeric values:
    - When present: show with 4 decimals, using dot (e.g., 14.1200)
    - When missing: show placeholder '00.0000'
    """
    v = st.session_state.get(value_key, None)
    value_str = fmt_num_dot(float(v)) if v is not None else ""
    st.text_input(
        label,
        value=value_str,
        key=f"tx_{value_key}",
        disabled=True,
        placeholder="00.00",
    )

#Reset
def _pop_state_keys(keys):
    for k in keys:
        st.session_state.pop(k, None)
        st.session_state.pop(f"tx_{k}", None)

def reset_app():
    feature_keys = list(ALL_LABELS.keys())
    meta_keys = list(meta_labels.keys())

    _pop_state_keys(feature_keys + meta_keys)

    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1



#UI Input Data
def page_input_data():
    st.title("Breast Cancer Detection")

    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0)

    uploaded = st.file_uploader(
        "Upload PDF report",
        type=["pdf"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    if uploaded:
        try:
            parsed_vals, parsed_meta = parse_pdf(uploaded)

            # Save numbers (only if parsed)
            for k, v in parsed_vals.items():
                if v is not None:
                    st.session_state[k] = v

            # Save meta
            for k, v in parsed_meta.items():
                st.session_state[k] = v

            found_count = sum(v is not None for v in parsed_vals.values())
            st.success(
                f"File uploaded successfully."
            )
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")

    cols = st.columns(2)
    with cols[0]:
        locked_text("Patient ID", "patient_id", placeholder="Patient ID")
    with cols[1]:
        locked_text("Date", "date", placeholder="Date")

    def render_section(title: str, keys: list[str], labels: dict[str, str]):
        st.subheader(title)
        for k in keys:
            locked_text_numeric(f"{labels[k]} :", k)

    render_section("Mean Parameters",  features_mean_params,  labels_mean_params)
    render_section("SE Parameters",    features_se_params,    labels_se_params)
    render_section("Worst Parameters", features_worst_params, labels_worst_params)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        do_predict = st.button("Predict", use_container_width=True)
    with bcol2:
        st.button("Reset", type="secondary", on_click=reset_app, use_container_width=True)

    if do_predict:
        if not uploaded:
            st.warning("⚠️ Please upload the PDF report first before prediction.")
        else:
            required_keys = features_mean_params + features_se_params + features_worst_params
            missing_keys = [k for k in required_keys if k not in st.session_state or st.session_state[k] is None]

            if missing_keys:
                # gabungkan semua labels
                all_labels = {}
                all_labels.update(labels_mean_params)
                all_labels.update(labels_se_params)
                all_labels.update(labels_worst_params)

                # buat list nama readable
                miss_params = [all_labels.get(k, k.replace("_", " ").title()) for k in missing_keys]

                st.error(
                    "❌ The uploaded PDF is missing required parameters:\n\n"
                    + ", ".join(miss_params)
                )
            else:
                with st.spinner("Predicting…"):
                    time.sleep(2)
                st.session_state.page = "hasil"
                st.rerun()



#Prediction Results Page
def getf(key: str, default: float = 0.0) -> float:
    v = st.session_state.get(key, None)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def display_prediction_result(predict_value: float):
    try:
        pv = float(np.asarray(predict_value).squeeze())
    except Exception:
        pv = float(predict_value)

    cls = 1 if pv >= 0.5 else 0
    confidence = pv if cls == 1 else (1 - pv)
    percent = confidence * 100.0
    label = {0: "Malignant", 1: "Benign"}[cls]

    st.markdown(
        f"""
        <div style="background-color:#f0f0f0;padding:20px;border-radius:10px;">
            <h2 style="color:#333;text-align:center;margin:0;">Prediction Results</h2>
            <p style="text-align:center;font-size:24px;margin:6px 0 0;">
                {percent:.2f}% {label}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _fmt_pretty_num(val, decimals=4):
    if val is None:
        return "-"
    try:
        s = f"{float(val):.{decimals}f}"
        s = s.rstrip("0").rstrip(".")
        if "." not in s:
            s = s + ".0"
        return s
    except Exception:
        return "-"

def render_params_table():
    st.markdown("""
    <style>
      .nice-wrap {
            margin-top: 1px;
            margin-bottom: 16px;
        }
      table.nice {
        width: 100%;
        border-collapse: collapse;
        border-spacing: 0;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        overflow: hidden;
      }
      .nice thead th {
        background: #DE2165; color: #fff; padding: 12px 16px;
        text-align: center; font-weight: 700;
      }
      .nice tbody td, .nice tbody th {
        padding: 12px 16px; border-bottom: 1px solid #e9e9e9;
      }
      .nice tbody tr:nth-child(even) { background: #f9f9f9; }
      .nice tbody tr:last-child td { border-bottom: 3px solid #DE2165; }
      .nice tbody th { text-align: left; font-weight: 600; color: #333; white-space: nowrap; }
      .nice td.num { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    rows_html = []
    for p in base_params:
        mean  = _fmt_pretty_num(st.session_state.get(f"{p}_mean"))
        se    = _fmt_pretty_num(st.session_state.get(f"{p}_se"))
        worst = _fmt_pretty_num(st.session_state.get(f"{p}_worst"))
        rows_html.append(f"""
        <tr>
          <th>{p.replace('_',' ').title()}</th>
          <td class="num">{mean}</td>
          <td class="num">{se}</td>
          <td class="num">{worst}</td>
        </tr>""")

    html_table = f"""
    <div class="nice-wrap">
      <table class="nice">
        <thead><tr><th>Parameters</th><th>Mean</th><th>SE</th><th>Worst</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>"""
    st.markdown(html_table, unsafe_allow_html=True)   
    
def page_hasil_input_data(predict_fn=None):
    st.subheader("Details Report Patient")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**Patient ID:** {st.session_state.get('patient_id', '-')}")
    with mc2:
        st.markdown(f"**Date:** {st.session_state.get('date', '-')}")

    #Table parameter display
    render_params_table()

    order = features_mean_params + features_se_params + features_worst_params
    features = np.array([[getf(k) for k in order]], dtype=np.float32)
    st.session_state["input_features"] = features  # keep for reuse

    _predictor = predict_fn or predict
    try:
        raw_pred = _predictor(features)
        # handle tf.Tensor / np.ndarray / dict outputs
        if isinstance(raw_pred, dict):
            raw_pred = next(iter(raw_pred.values()))
        raw_pred = getattr(raw_pred, "numpy", lambda: raw_pred)()
        prob = float(np.asarray(raw_pred).squeeze())
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    #Display prediction results
    display_prediction_result(prob)

    st.write("")
    if st.button("Done", key="goback", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        st.session_state.page = "input"
        st.rerun()

def app():
    #Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    #Display the current page based on session state
    if st.session_state.page == 'input':
        page_input_data()
    elif st.session_state.page == 'hasil':
        page_hasil_input_data()

if __name__ == '__main__':
    app()




