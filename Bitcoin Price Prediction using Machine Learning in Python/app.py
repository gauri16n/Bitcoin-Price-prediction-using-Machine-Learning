import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model

# Load model and scaler
@st.cache_resource
def load_models():
    model = load_model('models/lstm_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

st.title('🚀 Bitcoin Price Predictor')
st.write('Enter past 60 days Close prices to predict next day Bitcoin Close price using trained LSTM model.')

# Input for sequence
st.subheader('Input Sequence (60 past Close prices)')
seq = []
for i in range(60):
    val = st.number_input(f'Day {60-i}:', value=50000.0, step=100.0)
    seq.insert(0, val)

if st.button('Predict Next Close Price'):
    model, scaler = load_models()
    if len(seq) == 60:
        # Prepare input
        input_seq = np.array(seq).reshape(1, 60, 1)
        input_scaled = scaler.transform(np.array(seq).reshape(-1,1))
        input_scaled = input_scaled.reshape(1, 60, 1)
        
        # Predict
        pred_scaled = model.predict(input_scaled)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        st.success(f'**Predicted Next Day Close: ${pred_price:,.2f}**')
        
        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(range(60), seq, label='Past 60 Days')
        ax.plot(60, pred_price, 'ro', label='Prediction')
        ax.set_title('Bitcoin Price Prediction')
        ax.legend()
        st.pyplot(fig)
    else:
        st.error('Enter exactly 60 values.')

st.info('''**Note:** Train model first with `python bitcoin_prediction.py`, then use this app. Update app.py to load your trained scaler.''')
