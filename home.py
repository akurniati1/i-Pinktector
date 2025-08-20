import streamlit as st
from PIL import Image

def app():
    st.title("i-Pinktector")

    # Menambahkan CSS untuk mengubah latar belakang
    st.markdown(
        """
        <style>
        body {
            background-color: #FFC0CB; /* Warna pink */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Membagi layar menjadi dua kolom
    kolom_teks, kolom_gambar = st.columns(2)

    # Kolom Teks
    with kolom_teks:
        st.markdown(
            """
            Breast cancer is one of the most common cancers affecting women throughout the world. Early detection and accurate diagnosis are essential to increase survival rates. Machine Learning (ML) offers significant improvements in diagnostics, with Artificial Neural Networks (ANN) being particularly effective. To further improve accuracy, a novel hybrid method combining ANN and Matrix Factorization has been developed. Matrix factorization is applied in ANN which works on the weight matrix in the network to solve challenges such as slow convergence and high computational cost. i-Pinktector offers a web-based solution for detecting  types of breast cancer, both malignant and benign, using a hybrid approach that combines ANN and Matrix Factorization.
            """
        )

    with kolom_gambar:
        image = Image.open("logo.png")
        st.image(image, use_container_width=True)

    #df = pd.read_excel(
     #   io='data.xlsx',
      #  engine='openpyxl',
    #)

    #st.write(df)


if __name__ == '__main__':
    app()

