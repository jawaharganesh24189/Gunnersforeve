"""
PancakeSwap Prediction Bot - BNB 5-Minute Model

This module implements a BiLSTM + Attention model for predicting BNB price
direction on PancakeSwap's 5-minute prediction rounds. It applies Sequence
Modeling (S2) and Classification concepts from DLA study.

Architecture:
    - Input Layer: (Batch, 30, Features)
    - Encoder (BiLSTM): Captures momentum (price acceleration/exhaustion)
    - Attention Layer (MultiHeadAttention): Focuses on volume spikes (whale activity)
    - Output: Sigmoid (Probability of closing HIGHER than the Lock Price)

CRISP-DM Workflow:
    1. Business Understanding - PancakeSwap 5-min prediction game
    2. Data Understanding & Preparation - OHLCV + contract sentiment
    3. Modeling - BiLSTM + Attention
    4. Evaluation - Expected Value (EV) based trading logic
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
SEQ_LENGTH = 30       # Look back at last 30 minutes
PREDICT_AHEAD = 5     # Predict 5 minutes into the future
FEATURES = 5          # [Close, Volume, RSI, Bull_Ratio, Bear_Ratio]


# ==========================================
# 1. DATA GENERATOR (Simulating BNB Price & Contract Data)
# ==========================================
def generate_market_data(n_minutes=10000):
    """
    Simulates BNB price movement and Prediction Pool Sentiment.

    Generates synthetic 1-minute OHLCV data for BNB/USDT along with
    PancakeSwap pool sentiment (Bull/Bear payout multipliers).

    Args:
        n_minutes: Number of minutes of data to generate.

    Returns:
        DataFrame with columns: Close, Volume, Bull_Payout, Bear_Payout, RSI
    """
    prices = [300.0]
    volumes = [1000.0]
    bull_ratios = [1.8]  # Payout multiplier

    for _ in range(n_minutes):
        # Random Walk for Price
        change = np.random.normal(0, 1.5)  # Volatility
        new_price = prices[-1] + change
        prices.append(new_price)

        # Volume spikes on large moves
        vol = np.random.normal(1000, 200) + (abs(change) * 100)
        volumes.append(vol)

        # Pool Sentiment (Crowd usually chases the trend)
        # If price went up, crowd goes Bull (lowering Bull payout)
        sentiment = 1.8 - (change * 0.1)
        bull_ratios.append(np.clip(sentiment, 1.1, 3.0))

    df = pd.DataFrame({
        'Close': prices,
        'Volume': volumes,
        'Bull_Payout': bull_ratios,
        'Bear_Payout': [4.0 - x for x in bull_ratios]  # Approximate inverse
    })

    # Calculate RSI (Technical Indicator)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.fillna(50)

    return df


# ==========================================
# 2. PREPROCESSING (Sequence Creation)
# ==========================================
def create_sequences(df, seq_length=SEQ_LENGTH, predict_ahead=PREDICT_AHEAD):
    """
    Create input/output sequences for the model.

    Uses a sliding window of seq_length minutes to predict whether the
    price will be higher predict_ahead minutes in the future.

    Args:
        df: DataFrame with market data columns.
        seq_length: Number of past minutes to use as input.
        predict_ahead: Number of minutes ahead to predict.

    Returns:
        Tuple of (X, y, scaler) where X is the input array,
        y is the binary label array, and scaler is the fitted MinMaxScaler.
    """
    X = []
    y = []

    # Normalize Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create Windows
    data_val = df.values

    for i in range(seq_length, len(df) - predict_ahead):
        # Input: Past seq_length mins
        X.append(scaled_data[i - seq_length:i])

        # Target: Did price go UP in the next predict_ahead mins?
        current_price = data_val[i][0]  # Close is index 0
        future_price = data_val[i + predict_ahead][0]

        label = 1 if future_price > current_price else 0
        y.append(label)

    return np.array(X), np.array(y), scaler


# ==========================================
# 3. THE MODEL (BiLSTM + Attention)
# ==========================================
def build_pancake_model(seq_length=SEQ_LENGTH, features=FEATURES):
    """
    Build the BiLSTM + Attention model for price prediction.

    Architecture:
        - BiLSTM: Reads minute-by-minute price action to detect momentum
        - MultiHeadAttention: Focuses on volume spikes (whale activity)
        - GlobalAveragePooling + Dense: Makes the Bull/Bear decision

    Args:
        seq_length: Length of the input sequence (default: 30).
        features: Number of input features (default: 5).

    Returns:
        Compiled Keras Model with sigmoid output.
    """
    inputs = Input(shape=(seq_length, features))

    # Layer 1: BiLSTM (Momentum Detector)
    # Reads minute-by-minute price action
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)

    # Layer 2: Self-Attention (Whale Detector)
    # Focuses on specific minutes with abnormal volume
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Layer 3: Decision
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name="Bull_Probability")(x)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==========================================
# 4. TRADING STRATEGY (Inference & Odds Calculation)
# ==========================================
def trade_logic(model, current_sequence, bull_payout, bear_payout,
                seq_length=SEQ_LENGTH, features=FEATURES,
                confidence_threshold=0.60, ev_threshold=0.2):
    """
    Decides whether to bet based on Model Confidence AND Pool Odds (EV).

    Uses the Kelly Criterion / Expected Value approach:
        EV = (Probability * Payout) - (Probability_Loss * Stake)

    Only places a bet when both the confidence threshold and EV threshold
    are met, ensuring a positive expected value edge.

    Args:
        model: Trained Keras model.
        current_sequence: Input sequence array of shape (seq_length, features).
        bull_payout: Current Bull payout multiplier from the contract.
        bear_payout: Current Bear payout multiplier from the contract.
        seq_length: Sequence length (default: 30).
        features: Number of features (default: 5).
        confidence_threshold: Minimum probability to place a bet (default: 0.60).
        ev_threshold: Minimum expected value to place a bet (default: 0.2).

    Returns:
        Tuple of (decision, prob_bull, ev_bull, ev_bear) where decision is
        one of "BET BULL", "BET BEAR", or "SKIP".
    """
    # 1. Get AI Prediction
    seq_reshaped = current_sequence.reshape(1, seq_length, features)
    prob_bull = float(model.predict(seq_reshaped, verbose=0)[0][0])
    prob_bear = 1.0 - prob_bull

    decision = "SKIP"

    # 2. Calculate Expected Value (EV)
    # EV = (Probability * Payout) - (Probability_Loss * Stake)

    # Check Bull Case
    ev_bull = (prob_bull * bull_payout) - (prob_bear * 1)

    # Check Bear Case
    ev_bear = (prob_bear * bear_payout) - (prob_bull * 1)

    # 3. Thresholding (Only bet if we have an edge)
    if ev_bull > ev_threshold and prob_bull > confidence_threshold:
        decision = "BET BULL"
    elif ev_bear > ev_threshold and prob_bear > confidence_threshold:
        decision = "BET BEAR"

    return decision, prob_bull, ev_bull, ev_bear


def run_prediction_pipeline(n_minutes=10000, epochs=10, batch_size=32):
    """
    Run the full prediction pipeline: generate data, train model, simulate trade.

    This is the end-to-end CRISP-DM workflow for the PancakeSwap Prediction Bot.

    Args:
        n_minutes: Number of minutes of simulated market data.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Tuple of (model, history, trade_result) where trade_result is
        a dict with keys: action, confidence, ev_bull, ev_bear.
    """
    print(">>> FETCHING ON-CHAIN DATA (Simulated)...")
    market_data = generate_market_data(n_minutes)

    print(">>> PREPARING SEQUENCES...")
    X, y, scaler = create_sequences(market_data)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training Shapes: X={X_train.shape}, y={y_train.shape}")

    # Build and train model
    model = build_pancake_model()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Simulate a live round
    print("\n>>> LIVE ROUND PREDICTION ---")
    latest_data = X_test[-1]
    current_bull_payout = 1.95
    current_bear_payout = 1.95

    action, conf, ev_up, ev_down = trade_logic(
        model, latest_data, current_bull_payout, current_bear_payout
    )

    print(f"AI Bull Probability: {conf:.2%}")
    print(f"EV Bull: {ev_up:.2f} | EV Bear: {ev_down:.2f}")
    print(f"STRATEGY CALL: {action}")

    trade_result = {
        'action': action,
        'confidence': conf,
        'ev_bull': ev_up,
        'ev_bear': ev_down
    }

    return model, history, trade_result


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    run_prediction_pipeline()
