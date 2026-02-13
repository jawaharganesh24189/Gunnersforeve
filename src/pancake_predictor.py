"""
PancakeSwap Prediction Bot - BNB 5-Minute Model

This module implements two models for predicting BNB price direction on
PancakeSwap's 5-minute prediction rounds. It applies Sequence Modeling (S2)
and Classification concepts from DLA study.

Models:
    1. Base Model (BiLSTM + Attention):
        - BiLSTM encoder for momentum detection
        - MultiHeadAttention for volume spike focus
        - Sigmoid output for Bull probability

    2. Robust Model (Conv1D + Stacked BiLSTM + Attention + Residual):
        - GaussianNoise for overfitting prevention
        - Conv1D for automatic candlestick pattern extraction
        - Stacked BiLSTMs for fast and deep temporal patterns
        - MultiHeadAttention with residual connection (ResNet-style)
        - Sigmoid output for Bull probability

CRISP-DM Workflow:
    1. Business Understanding - PancakeSwap 5-min prediction game
    2. Data Understanding & Preparation - OHLCV + contract sentiment
    3. Modeling - BiLSTM + Attention (base) and Conv1D + BiLSTM + Attention (robust)
    4. Evaluation - Expected Value (EV) based trading logic, model comparison, ensemble
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Conv1D, GaussianNoise, Add, LayerNormalization, GlobalAveragePooling1D
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
# 3b. ROBUST MODEL (Conv1D + Stacked BiLSTM + Attention + Residual)
# ==========================================
def build_robust_pancake_model(seq_length=SEQ_LENGTH, features=FEATURES):
    """
    Build a robust Conv1D + Stacked BiLSTM + Attention model with residual connections.

    Architecture:
        - GaussianNoise: Prevents overfitting by injecting noise during training
        - Conv1D: Automatically learns candlestick patterns (3-minute kernel)
        - Stacked BiLSTMs: Fast patterns (layer 1) and deep trends (layer 2)
        - MultiHeadAttention + Residual: Correlations with ResNet-style skip connection
        - GlobalAveragePooling + Dense: Makes the Bull/Bear decision

    Args:
        seq_length: Length of the input sequence (default: 30).
        features: Number of input features (default: 5).

    Returns:
        Compiled Keras Model with sigmoid output.
    """
    inputs = Input(shape=(seq_length, features))

    # --- LAYER 1: ROBUSTNESS (Gaussian Noise) ---
    # We inject random noise (stddev=0.05) to the input data.
    # This prevents the model from memorizing exact prices (Overfitting).
    # It learns to see the "Shape" through the "Fog".
    # Note: GaussianNoise is only active during training (automatically
    # disabled during inference via model.predict()).
    x = GaussianNoise(0.05)(inputs)

    # --- LAYER 2: FEATURE EXTRACTION (Conv1D) ---
    # Filters=32, Kernel=3 means "Look at 3 minutes at a time".
    # This automatically learns candlestick patterns (e.g., Engulfing candles).
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)  # Keeps values stable

    # --- LAYER 3: TEMPORAL MEMORY (Stacked BiLSTMs) ---
    # We stack two LSTMs.
    # LSTM 1: Fast patterns (return_sequences=True keeps the timeline)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # LSTM 2: Deep patterns (The "Trend")
    # We save this output 'lstm_out' for the Residual connection later
    lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # --- LAYER 4: ATTENTION + RESIDUAL (The Transformer Block) ---
    # Multi-Head Attention looks for correlations across the 30-minute window
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)

    # RESIDUAL CONNECTION (Add & Norm)
    # We add the LSTM memory (lstm_out) to the Attention insight (attn_out).
    # This is the "Safety Net" that creates robust Deep Learning models (like ResNet).
    x = Add()([lstm_out, attn_out])
    x = LayerNormalization()(x)

    # --- LAYER 5: DECISION ---
    x = GlobalAveragePooling1D()(x)  # Summarize the whole sequence

    # Dense layers to reason about the features
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # Heavy dropout for financial data

    output = layers.Dense(1, activation='sigmoid', name="Bull_Probability")(x)

    model = models.Model(inputs=inputs, outputs=output)

    # Use a lower learning rate for robust fine-tuning
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ==========================================
# 3c. ENSEMBLE PREDICTION (Dual Model)
# ==========================================
def ensemble_predict(base_model, robust_model, sequence,
                     seq_length=SEQ_LENGTH, features=FEATURES,
                     base_weight=0.5, robust_weight=0.5):
    """
    Generate predictions from both models and combine them.

    Uses a weighted average of both model predictions. The ensemble
    reduces variance and can improve accuracy by combining the strengths
    of both architectures.

    Args:
        base_model: Trained base BiLSTM + Attention model.
        robust_model: Trained robust Conv1D + BiLSTM + Attention model.
        sequence: Input sequence of shape (seq_length, features).
        seq_length: Sequence length (default: 30).
        features: Number of features (default: 5).
        base_weight: Weight for base model prediction (default: 0.5).
        robust_weight: Weight for robust model prediction (default: 0.5).

    Returns:
        Dict with keys: base_prob, robust_prob, ensemble_prob,
        base_decision, robust_decision, ensemble_decision.
    """
    seq_reshaped = sequence.reshape(1, seq_length, features)

    base_prob = float(base_model.predict(seq_reshaped, verbose=0)[0][0])
    robust_prob = float(robust_model.predict(seq_reshaped, verbose=0)[0][0])
    ensemble_prob = (base_weight * base_prob) + (robust_weight * robust_prob)

    def _decision(prob):
        if prob > 0.60:
            return "BET BULL"
        elif prob < 0.40:
            return "BET BEAR"
        return "SKIP"

    return {
        'base_prob': base_prob,
        'robust_prob': robust_prob,
        'ensemble_prob': ensemble_prob,
        'base_decision': _decision(base_prob),
        'robust_decision': _decision(robust_prob),
        'ensemble_decision': _decision(ensemble_prob),
    }


# ==========================================
# 3d. KNOWLEDGE DISTILLATION (Teacher-Student)
# ==========================================
def build_distilled_model(teacher_base, teacher_robust, X_train,
                          seq_length=SEQ_LENGTH, features=FEATURES,
                          epochs=5, batch_size=32):
    """
    Build a student model that learns from both teacher models (knowledge distillation).

    The student model is trained on soft labels (averaged predictions from
    both teachers), which transfers the learned knowledge of both
    architectures into a single smaller model.

    Args:
        teacher_base: Trained base model.
        teacher_robust: Trained robust model.
        X_train: Training input data.
        seq_length: Sequence length (default: 30).
        features: Number of features (default: 5).
        epochs: Number of training epochs (default: 5).
        batch_size: Training batch size (default: 32).

    Returns:
        Trained student model.
    """
    # Generate soft labels from both teachers
    base_preds = teacher_base.predict(X_train, verbose=0)
    robust_preds = teacher_robust.predict(X_train, verbose=0)
    soft_labels = (base_preds + robust_preds) / 2.0

    # Build a lightweight student model
    inputs = Input(shape=(seq_length, features))
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name="Bull_Probability")(x)

    student = models.Model(inputs=inputs, outputs=output)
    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train on soft labels from teachers
    student.fit(X_train, soft_labels, epochs=epochs, batch_size=batch_size, verbose=1)

    return student


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
