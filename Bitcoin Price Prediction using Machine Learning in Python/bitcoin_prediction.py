import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('bitcoin.csv', parse_dates=['Date'], index_col='Date')
df = df.sort_index()
df = df[['Close']].dropna()

# Scale
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Save scaler for Streamlit app
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
print('Scaler saved to models/scaler.pkl')

# Create sequences (60 days lookback)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split 80/20
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Predict
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse scale
train_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(train_pred),1)), train_pred), axis=1))[:,1]
test_pred = scaler.inverse_transform(np.concatenate((np.zeros((len(test_pred),1)), test_pred), axis=1))[:,1]
y_train_inv = scaler.inverse_transform(np.concatenate((np.zeros((len(y_train),1)), y_train.reshape(-1,1)), axis=1))[:,1]
y_test_inv = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test),1)), y_test.reshape(-1,1)), axis=1))[:,1]

# RMSE
rmse_test = np.sqrt(mean_squared_error(y_test_inv, test_pred))
print(f'Test RMSE: {rmse_test:.2f}')

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index[seq_length:split+seq_length], y_train_inv, label='Train Actual')
plt.plot(df.index[seq_length:split+seq_length], train_pred, label='Train Pred')
plt.plot(df.index[split+seq_length:], y_test_inv, label='Test Actual')
plt.plot(df.index[split+seq_length:], test_pred, label='Test Pred')
plt.legend()
plt.title('Bitcoin Close Price Prediction')
plt.show()

# Save model
model.save('models/lstm_model.keras')
print('Model saved to models/lstm_model.h5')
