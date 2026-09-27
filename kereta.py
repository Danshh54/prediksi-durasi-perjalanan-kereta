import streamlit as st
import sklearn
import joblib, os
import numpy as np 

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    st.title("Prediksi Durasi Perjalanan Kereta Api Menggunakan Regresi Linier")

    jumlah_stasiun = st.slider("Berapa jumlah stasiun pemberhentian?", 0, 30)

    if st.button("Proses"):
        regressor = load_prediction_model("linear_regression_kereta.pkl")
        
        input_reshaped = np.array(jumlah_stasiun).reshape(-1, 1)
        predicted_duration = regressor.predict(input_reshaped)

        st.info("Estimasi durasi perjalanan untuk kereta dengan {} stasiun pemberhentian: {} menit".format(jumlah_stasiun, (predicted_duration[0][0].round(2))))

if __name__ == '__main__':
    main()