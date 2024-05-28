import streamlit as st
import numpy as np
import tensorflow as tf
import time
import zipfile
import os

# Replace 'file.zip' with the name of your ZIP file
zip_file_path = 'C-Tr-SVD.zip'
extract_to_directory = 'C-Tr-SVD'

# Extract the ZIP file
if not os.path.exists(extract_to_directory):
    os.makedirs(extract_to_directory)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_directory)

# Load the trained model using TFSMLayer
class SVDLayer2(tf.keras.layers.Layer):
    def __init__(self, weight, type_svd, rank, activation=None, **kwargs):
        super(SVDLayer2, self).__init__(**kwargs)
        self.U, self.S, self.V = self.svd_decomposition(weight, type_svd, rank)
        self.activation = tf.keras.activations.get(activation)

    def svd_decomposition(self, weight, type_svd, rank):
        # Placeholder for the actual SVD decomposition logic
        # Replace with the actual implementation for Ur, Sr, Vr
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

def predict(input_features):
    prediction = model(input_features)
    return prediction

def page_input_data():
    st.title("Breast Cancer Detection")
    st.subheader("Mean Parameter")
    radius_mean = st.number_input("Radius Mean: ", 0.0, step=0.01, format="%.2f")
    texture_mean = st.number_input("Texture Mean: ", 0.0, step=0.01, format="%.2f")
    perimeter_mean = st.number_input("Perimeter Mean: ", 0.0, step=0.01, format="%.2f")
    area_mean = st.number_input("Area Mean: ", 0.0, step=0.01, format="%.2f")
    smoothness_mean = st.number_input("Smoothness Mean: ", 0.0, step=0.01, format="%.2f")
    compactness_mean = st.number_input("Compactness Mean: ", 0.0, step=0.01, format="%.2f")
    concavity_mean = st.number_input("Concavity Mean: ", 0.0, step=0.01, format="%.2f")
    concave_points_mean = st.number_input("Concave Points Mean: ", 0.0, step=0.01, format="%.2f")
    symmetry_mean = st.number_input("Symmetry Mean: ", 0.0, step=0.01, format="%.2f")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean: ", 0.0, step=0.01, format="%.2f")

    st.subheader("SE Parameter")
    radius_se = st.number_input("Radius SE: ", 0.0, step=0.01, format="%.2f")
    texture_se = st.number_input("Texture SE: ", 0.0, step=0.01, format="%.2f")
    perimeter_se = st.number_input("Perimeter SE: ", 0.0, step=0.01, format="%.2f")
    area_se = st.number_input("Area SE: ", 0.0, step=0.01, format="%.2f")
    smoothness_se = st.number_input("Smoothness SE: ", 0.0, step=0.01, format="%.2f")
    compactness_se = st.number_input("Compactness SE: ", 0.0, step=0.01, format="%.2f")
    concavity_se = st.number_input("Concavity SE: ", 0.0, step=0.01, format="%.2f")
    concave_points_se = st.number_input("Concave Points SE: ", 0.0, step=0.01, format="%.2f")
    symmetry_se = st.number_input("Symmetry SE: ", 0.0, step=0.01, format="%.2f")
    fractal_dimension_se = st.number_input("Fractal Dimension SE: ", 0.0, step=0.01, format="%.2f")

    st.subheader("Worst Parameter")
    radius_worst = st.number_input("Radius Worst: ", 0.0, step=0.01, format="%.2f")
    texture_worst = st.number_input("Texture Worst: ", 0.0, step=0.01, format="%.2f")
    perimeter_worst = st.number_input("Perimeter Worst: ", 0.0, step=0.01, format="%.2f")
    area_worst = st.number_input("Area Worst: ", 0.0, step=0.01, format="%.2f")
    smoothness_worst = st.number_input("Smoothness Worst: ", 0.0, step=0.01, format="%.2f")
    compactness_worst = st.number_input("Compactness Worst: ", 0.0, step=0.01, format="%.2f")
    concavity_worst = st.number_input("Concavity Worst: ", 0.0, step=0.01, format="%.2f")
    concave_points_worst = st.number_input("Concave Points Worst: ", 0.0, step=0.01, format="%.2f")
    symmetry_worst = st.number_input("Symmetry Worst: ", 0.0, step=0.01, format="%.2f")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst: ", 0.0, step=0.01, format="%.2f")

    if st.button('Predict'):
        with st.spinner('Predicting...'):
            time.sleep(2)

            input_features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                        smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                                        fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                                        smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                                        symmetry_worst, fractal_dimension_worst]])

            # Save data to session state
            st.session_state['input_features'] = input_features

            # Switch to the result page
            st.session_state.page = 'hasil'
            st.experimental_rerun()

def display_prediction_result(predict_value):
    # Mengonversi nilai prediksi menjadi persentase
    percent = np.where(predict_value >= 0.5, predict_value * 100, (1 - predict_value) * 100)
    # Mendefinisikan label kelas
    type_prediction = {0: 'Malignant', 1: 'Benign'}
    predicted_class = 1 if predict_value > 0.5 else 0
    # Membuat tampilan dengan desain menarik
    st.markdown(
        f"""
        <div style="background-color:#f0f0f0;padding:20px;border-radius:10px;">
            <h2 style="color:#333;text-align:center;">Prediction Results</h2>
            <p style="text-align:center;font-size:24px;"> {percent:.2f}% {type_prediction.get(predicted_class, "Unknown")} </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def page_hasil_input_data():
    st.subheader('Details Data Entry')

    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css');
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 18px;
        text-align: left;
    }
    .data-table th, .data-table td {
        padding: 12px 15px;
    }
    .data-table th {
        background-color: #DE2165;
        color: #ffffff;
        text-align: center;
    }
    .data-table tr {
        border-bottom: 1px solid #dddddd;
    }
    .data-table tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .data-table tr:last-of-type {
        border-bottom: 2px solid #DE2165;
    }
    .data-table td {
        text-align: center;
    }
    .title-with-icon {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .title-with-icon i {
        color: #DE2165;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the table using markdown
    table_html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Parameters</th>
                <th>Mean</th>
                <th>SE</th>
                <th>Worst</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Radius</td>
                <td>{radius_mean}</td>
                <td>{radius_se}</td>
                <td>{radius_worst}</td>
            </tr>
            <tr>
                <td>Texture</td>
                <td>{texture_mean}</td>
                <td>{texture_se}</td>
                <td>{texture_worst}</td>
            </tr>
            <tr>
                <td>Perimeter</td>
                <td>{perimeter_mean}</td>
                <td>{perimeter_se}</td>
                <td>{perimeter_worst}</td>
            </tr>
            <tr>
                <td>Area</td>
                <td>{area_mean}</td>
                <td>{area_se}</td>
                <td>{area_worst}</td>
            </tr>
            <tr>
                <td>Smoothness</td>
                <td>{smoothness_mean}</td>
                <td>{smoothness_se}</td>
                <td>{smoothness_worst}</td>
            </tr>
            <tr>
                <td>Compactness</td>
                <td>{compactness_mean}</td>
                <td>{compactness_se}</td>
                <td>{compactness_worst}</td>
            </tr>
            <tr>
                <td>Concavity</td>
                <td>{concavity_mean}</td>
                <td>{concavity_se}</td>
                <td>{concavity_worst}</td>
            </tr>
            <tr>
                <td>Concave Points</td>
                <td>{concave_mean}</td>
                <td>{concave_se}</td>
                <td>{concave_worst}</td>
            </tr>
            <tr>
                <td>Symmetry</td>
                <td>{symmetry_mean}</td>
                <td>{symmetry_se}</td>
                <td>{symmetry_worst}</td>
            </tr>
            <tr>
                <td>Fractal Dimension</td>
                <td>{fractal_dimension_mean}</td>
                <td>{fractal_dimension_se}</td>
                <td>{fractal_dimension_worst}</td>
            </tr>
        </tbody>
    </table>
    """.format(
        radius_mean=st.session_state.get('radius_mean', '0.0'),
        texture_mean=st.session_state.get('texture_mean', '0.0'),
        perimeter_mean=st.session_state.get('perimeter_mean', '0.0'),
        area_mean=st.session_state.get('area_mean', '0.0'),
        smoothness_mean=st.session_state.get('smoothness_mean', '0.0'),
        compactness_mean=st.session_state.get('compactness_mean', '0.0'),
        concavity_mean=st.session_state.get('concavity_mean', '0.0'),
        concave_mean=st.session_state.get('concave_mean', '0.0'),
        symmetry_mean=st.session_state.get('symmetry_mean', '0.0'),
        fractal_dimension_mean=st.session_state.get('fractal_dimension_mean', '0.0'),
        radius_se=st.session_state.get('radius_se', '0.0'),
        texture_se=st.session_state.get('texture_se', '0.0'),
        perimeter_se=st.session_state.get('perimeter_se', '0.0'),
        area_se=st.session_state.get('area_se', '0.0'),
        smoothness_se=st.session_state.get('smoothness_se', '0.0'),
        compactness_se=st.session_state.get('compactness_se', '0.0'),
        concavity_se=st.session_state.get('concavity_se', '0.0'),
        concave_se=st.session_state.get('concave_se', '0.0'),
        symmetry_se=st.session_state.get('symmetry_se', '0.0'),
        fractal_dimension_se=st.session_state.get('fractal_dimension_se', '0.0'),
        radius_worst=st.session_state.get('radius_worst', '0.0'),
        texture_worst=st.session_state.get('texture_worst', '0.0'),
        perimeter_worst=st.session_state.get('perimeter_worst', '0.0'),
        area_worst=st.session_state.get('area_worst', '0.0'),
        smoothness_worst=st.session_state.get('smoothness_worst', '0.0'),
        compactness_worst=st.session_state.get('compactness_worst', '0.0'),
        concavity_worst=st.session_state.get('concavity_worst', '0.0'),
        concave_worst=st.session_state.get('concave_worst', '0.0'),
        symmetry_worst=st.session_state.get('symmetry_worst', '0.0'),
        fractal_dimension_worst=st.session_state.get('fractal_dimension_worst', '0.0'),
    )

    st.markdown(table_html, unsafe_allow_html=True)

    input_features = st.session_state.get('input_features')
    prediction = predict(input_features)
    predict_value = prediction['dense_8'].numpy ().flatten()[0]

    # Menampilkan hasil prediksi dengan desain yang menarik
    display_prediction_result(predict_value)

    # Menambahkan tombol "Go Back" dengan desain yang menarik
    st.markdown(
        """
        <style>
        .btn-goback {
            display: inline-block;
            padding: 8px 20px;
            font-size: 16px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            outline: none;
            color: #fff;
            background-color: #007bff;
            border: none;
            border-radius: 5px;
            box-shadow: 0 2px #0069d9;
        }

        .btn-goback:hover {
            background-color: #0056b3;
        }

        .btn-goback:active {
            background-color: #0056b3;
            box-shadow: 0 2px #0056b3;
            transform: translateY(1px);
        }
        </style>
        """
    , unsafe_allow_html=True)

    if st.button("Done", key="goback"):
        st.session_state.page = 'input'
        st.experimental_rerun()

def app():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    # Display the current page based on session state
    if st.session_state.page == 'input':
        page_input_data()
    elif st.session_state.page == 'hasil':
        page_hasil_input_data()

if __name__ == '__main__':
    app()
