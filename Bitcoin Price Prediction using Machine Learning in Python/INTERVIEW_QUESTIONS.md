# Interview/Teacher Questions for Bitcoin Price Prediction Project

## ML Model Used: **LSTM (Long Short-Term Memory)**

**Full Details:**
- **Type:** Deep Learning, Recurrent Neural Network (RNN) variant
- **Purpose:** Time series forecasting (sequential data)
- **Key Innovation:** 3 Gates (Forget, Input, Output Cell) - remembers long-term patterns, avoids vanishing gradients
- **Structure:** Cell state (memory highway), hidden state, gates control info flow
- **Architecture in project:** 
  - Input: (batch, 60 timesteps, 1 feature - Close price)
  - Layer1: LSTM(50, return_seq=True) + Dropout(0.2)
  - Layer2: LSTM(50) + Dropout(0.2)
  - Output: Dense(1) - next price
  - Optimizer: Adam, Loss: MSE
- **Why for Bitcoin:** Captures volatility, trends, cycles in price history

**Equations (simplified):**
- Forget gate: f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
- Update cell: C_t = f_t · C_{t-1} + i_t · tanh(W_C · [h_{t-1}, x_t] + b_C)

## Common Questions & Answers

### 1. **Why LSTM over other models (Random Forest/ARIMA)?**
**Answer:** RF not sequential, ARIMA linear. LSTM captures long dependencies/non-linearity in volatile crypto prices.

### 2. **LSTM architecture?**
**Answer:** 2 LSTM layers (50 units), Dropout 0.2, Dense(1). Seq: (batch, 60, 1).

### 3. **Hyperparameters?**
**Answer:** Seq length 60, epochs 50, batch 32, Adam, MSE loss.

### 4. **Metrics? Other evals?**
**Answer:** RMSE/MAE. Could add MAPE, directional accuracy.

### 5. **Feature engineering?**
**Answer:** Close price only; could lag features, returns, volume.

### 6. **Train/test split?**
**Answer:** 80/20 time-based (no future leak).

### 7. **Overfitting?**
**Answer:** Dropout, val_split. Monitor loss curves.

### 8. **Productionize?**
**Answer:** Streamlit app loads model, API (FastAPI), retrain scheduler.

**Demo:** `run_app.bat` → predict live!

Perfect for ML interviews.
