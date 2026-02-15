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

# PancakeSwap Prediction V2 Contract (BSC Mainnet)
PANCAKE_PREDICTION_ADDRESS = '0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA'

# PancakeSwap V2 Router and Factory (BSC Mainnet)
PANCAKE_ROUTER_ADDRESS = '0x10ED43C718714eb63d5aA57B78B54704E256024E'
PANCAKE_FACTORY_ADDRESS = '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'

# BNB and USDT token addresses on BSC
WBNB_ADDRESS = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c'
USDT_ADDRESS = '0x55d398326f99059fF775485246999027B3197955'

# Minimal ABI for the PancakeSwap Prediction contract (read-only)
PREDICTION_ABI = [
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "rounds",
        "outputs": [
            {"internalType": "uint256", "name": "epoch", "type": "uint256"},
            {"internalType": "uint256", "name": "startTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "lockTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "closeTimestamp", "type": "uint256"},
            {"internalType": "int256", "name": "lockPrice", "type": "int256"},
            {"internalType": "int256", "name": "closePrice", "type": "int256"},
            {"internalType": "uint256", "name": "lockOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "closeOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "totalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bullAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bearAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardBaseCalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardAmount", "type": "uint256"},
            {"internalType": "bool", "name": "oracleCalled", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Minimal ABI for PancakeSwap V2 Pair (for getting reserves and price)
PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Minimal ABI for PancakeSwap V2 Factory (to get pair address)
FACTORY_ABI = [
    {
        "constant": True,
        "inputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "name": "getPair",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]


# ==========================================
# 1a. LIVE DATA: PancakeSwap DEX Price via Web3
# ==========================================
def fetch_pancakeswap_price(rpc_url='https://bsc-dataseed1.binance.org/'):
    """
    Fetch real-time BNB/USDT price from PancakeSwap V2 liquidity pool.

    Queries the PancakeSwap BNB/USDT pair contract to get current reserves
    and calculates the spot price from the constant product formula (x * y = k).

    Args:
        rpc_url: BSC RPC endpoint URL.

    Returns:
        Dict with keys: price (BNB in USDT), reserve_bnb, reserve_usdt, timestamp.

    Raises:
        Exception: If the RPC call fails.
    """
    from web3 import Web3
    import time

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    # Get the pair address for WBNB/USDT
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(PANCAKE_FACTORY_ADDRESS),
        abi=FACTORY_ABI
    )
    
    pair_address = factory.functions.getPair(
        Web3.to_checksum_address(WBNB_ADDRESS),
        Web3.to_checksum_address(USDT_ADDRESS)
    ).call()
    
    # Get reserves from the pair contract
    pair = w3.eth.contract(
        address=Web3.to_checksum_address(pair_address),
        abi=PAIR_ABI
    )
    
    reserves = pair.functions.getReserves().call()
    token0 = pair.functions.token0().call()
    
    # Determine which reserve is BNB and which is USDT
    if token0.lower() == WBNB_ADDRESS.lower():
        reserve_bnb = reserves[0] / 1e18  # WBNB has 18 decimals
        reserve_usdt = reserves[1] / 1e18  # USDT has 18 decimals
    else:
        reserve_bnb = reserves[1] / 1e18
        reserve_usdt = reserves[0] / 1e18
    
    # Calculate price: USDT per BNB
    price = reserve_usdt / reserve_bnb if reserve_bnb > 0 else 0
    
    return {
        'price': price,
        'reserve_bnb': reserve_bnb,
        'reserve_usdt': reserve_usdt,
        'timestamp': time.time()
    }


def fetch_pancakeswap_ohlcv(timeframe='1m', limit=500, rpc_url='https://bsc-dataseed1.binance.org/'):
    """
    Fetch historical OHLCV data by sampling PancakeSwap prices at regular intervals.
    
    This function builds OHLCV candles by:
    1. Fetching historical block data from BSC
    2. Querying PancakeSwap pair reserves at each block
    3. Calculating prices and constructing OHLCV candles
    
    Note: This samples prices from chain state. For production, consider using
    The Graph Protocol subgraph for more efficient historical data access.

    Args:
        timeframe: Candle interval (default: '1m' for 1-minute candles).
        limit: Number of candles to fetch (default: 500).
        rpc_url: BSC RPC endpoint URL.

    Returns:
        DataFrame with columns: Timestamp, Open, High, Low, Close, Volume.
    """
    from web3 import Web3
    import time
    
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    # Get the pair address
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(PANCAKE_FACTORY_ADDRESS),
        abi=FACTORY_ABI
    )
    
    pair_address = factory.functions.getPair(
        Web3.to_checksum_address(WBNB_ADDRESS),
        Web3.to_checksum_address(USDT_ADDRESS)
    ).call()
    
    pair = w3.eth.contract(
        address=Web3.to_checksum_address(pair_address),
        abi=PAIR_ABI
    )
    
    # Determine token order
    token0 = pair.functions.token0().call()
    is_bnb_token0 = token0.lower() == WBNB_ADDRESS.lower()
    
    # BSC block time is ~3 seconds, so for 1-minute candles we need ~20 blocks per candle
    timeframe_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}.get(timeframe, 60)
    blocks_per_candle = max(1, int(timeframe_seconds / 3))
    
    current_block = w3.eth.block_number
    start_block = current_block - (blocks_per_candle * limit)
    
    print(f'>>> Fetching {limit} {timeframe} candles from PancakeSwap...')
    print(f'    Sampling blocks {start_block} to {current_block}')
    
    candles = []
    prices_in_window = []
    volume_in_window = 0
    window_start_time = None
    last_reserve_bnb = None
    
    for block_num in range(start_block, current_block + 1, max(1, blocks_per_candle // 20)):
        try:
            # Get block timestamp
            block = w3.eth.get_block(block_num)
            block_time = pd.to_datetime(block['timestamp'], unit='s')
            
            # Get reserves at this block
            reserves = pair.functions.getReserves().call(block_identifier=block_num)
            
            # Calculate price
            if is_bnb_token0:
                reserve_bnb = reserves[0] / 1e18
                reserve_usdt = reserves[1] / 1e18
            else:
                reserve_bnb = reserves[1] / 1e18
                reserve_usdt = reserves[0] / 1e18
            
            price = reserve_usdt / reserve_bnb if reserve_bnb > 0 else 0
            
            # Calculate volume (change in reserves)
            if last_reserve_bnb is not None:
                volume = abs(reserve_bnb - last_reserve_bnb)
            else:
                volume = 0
            last_reserve_bnb = reserve_bnb
            
            # Initialize window if needed
            if window_start_time is None:
                window_start_time = block_time
            
            # Check if we need to close this candle
            if (block_time - window_start_time).total_seconds() >= timeframe_seconds:
                if prices_in_window:
                    candles.append({
                        'Timestamp': window_start_time,
                        'Open': prices_in_window[0],
                        'High': max(prices_in_window),
                        'Low': min(prices_in_window),
                        'Close': prices_in_window[-1],
                        'Volume': volume_in_window
                    })
                
                # Start new window
                prices_in_window = [price]
                volume_in_window = volume
                window_start_time = block_time
            else:
                prices_in_window.append(price)
                volume_in_window += volume
                
        except Exception as e:
            # Skip blocks that fail (might be too old or API limits)
            continue
    
    # Close last candle
    if prices_in_window:
        candles.append({
            'Timestamp': window_start_time,
            'Open': prices_in_window[0],
            'High': max(prices_in_window),
            'Low': min(prices_in_window),
            'Close': prices_in_window[-1],
            'Volume': volume_in_window
        })
    
    df = pd.DataFrame(candles)
    if len(df) > 0:
        df.set_index('Timestamp', inplace=True)
        # Limit to requested number of candles
        df = df.tail(limit)
    
    print(f'    Received {len(df)} candles from PancakeSwap DEX')
    
    return df


# ==========================================
# 1b. LIVE DATA: PancakeSwap Contract via Web3
# ==========================================
def fetch_contract_data(rpc_url='https://bsc-dataseed1.binance.org/'):
    """
    Fetch currentEpoch, lockPrice, bullAmount, and bearAmount from the
    PancakeSwap Prediction V2 Smart Contract on BSC.

    Args:
        rpc_url: BSC RPC endpoint URL. Public endpoints:
            - https://bsc-dataseed1.binance.org/
            - https://bsc-dataseed2.binance.org/
            - https://bsc-dataseed3.binance.org/

    Returns:
        Dict with keys: epoch, lock_price, bull_amount, bear_amount,
        total_amount, bull_payout, bear_payout.

    Raises:
        ImportError: If web3 is not installed.
        Exception: If the RPC call fails.
    """
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(PANCAKE_PREDICTION_ADDRESS),
        abi=PREDICTION_ABI
    )

    epoch = contract.functions.currentEpoch().call()
    round_data = contract.functions.rounds(epoch).call()

    # round_data fields: epoch, startTs, lockTs, closeTs, lockPrice, closePrice,
    #   lockOracleId, closeOracleId, totalAmount, bullAmount, bearAmount,
    #   rewardBaseCalAmount, rewardAmount, oracleCalled
    total_amount = round_data[8]
    bull_amount = round_data[9]
    bear_amount = round_data[10]
    lock_price = round_data[4]

    # Calculate payouts (avoid division by zero)
    if bull_amount > 0:
        bull_payout = total_amount / bull_amount
    else:
        bull_payout = 1.0

    if bear_amount > 0:
        bear_payout = total_amount / bear_amount
    else:
        bear_payout = 1.0

    return {
        'epoch': epoch,
        'lock_price': lock_price / 1e8,  # Oracle price in 8 decimals
        'bull_amount': bull_amount / 1e18,  # BNB in wei
        'bear_amount': bear_amount / 1e18,
        'total_amount': total_amount / 1e18,
        'bull_payout': bull_payout,
        'bear_payout': bear_payout,
    }


# ==========================================
# 1c. COMBINED LIVE DATA PIPELINE
# ==========================================
def fetch_live_market_data(timeframe='1m', limit=500,
                           rpc_url='https://bsc-dataseed1.binance.org/',
                           use_contract=True):
    """
    Fetch real-time market data from PancakeSwap DEX and Prediction contract.

    Fetches OHLCV candles from PancakeSwap DEX via Web3, calculates RSI, and
    fetches Bull/Bear pool data from the PancakeSwap Prediction smart contract.

    Args:
        timeframe: Candle interval (default: '1m').
        limit: Number of candles (default: 500).
        rpc_url: BSC RPC endpoint for contract calls.
        use_contract: Whether to fetch on-chain prediction contract data (default: True).

    Returns:
        Tuple of (market_df, contract_info) where market_df has columns
        [Close, Volume, Bull_Payout, Bear_Payout, RSI] and contract_info
        is a dict with epoch/payout data (or None if contract fetch is skipped).
    """
    # 1. Fetch PancakeSwap DEX OHLCV data
    print(f'>>> Fetching {limit} {timeframe} candles from PancakeSwap DEX...')
    ohlcv_df = fetch_pancakeswap_ohlcv(timeframe=timeframe, limit=limit, rpc_url=rpc_url)
    
    if len(ohlcv_df) == 0:
        raise Exception("Failed to fetch OHLCV data from PancakeSwap. Please check RPC connection.")
    
    print(f'    Received {len(ohlcv_df)} candles, latest: {ohlcv_df.index[-1]}')

    # 2. Fetch Prediction contract data
    contract_info = None
    bull_payout = 1.95  # Default
    bear_payout = 1.95

    if use_contract:
        try:
            print(f'>>> Fetching PancakeSwap Prediction contract data from BSC...')
            contract_info = fetch_contract_data(rpc_url=rpc_url)
            bull_payout = contract_info['bull_payout']
            bear_payout = contract_info['bear_payout']
            print(f'    Epoch: {contract_info["epoch"]}, '
                  f'Bull Pool: {contract_info["bull_amount"]:.2f} BNB, '
                  f'Bear Pool: {contract_info["bear_amount"]:.2f} BNB')
            print(f'    Bull Payout: {bull_payout:.2f}x, Bear Payout: {bear_payout:.2f}x')
        except Exception as e:
            print(f'    WARNING: Prediction contract fetch failed ({e}). Using default payouts.')

    # 3. Build feature DataFrame
    df = pd.DataFrame({
        'Close': ohlcv_df['Close'].values,
        'Volume': ohlcv_df['Volume'].values,
    })

    # Calculate per-candle pool sentiment from price momentum
    price_changes = df['Close'].diff().fillna(0)
    sentiment = 1.8 - (price_changes * 0.01)
    df['Bull_Payout'] = np.clip(sentiment, 1.1, 3.0)
    df['Bear_Payout'] = 4.0 - df['Bull_Payout']

    # Override last row with actual contract payouts if available
    if contract_info is not None:
        df.loc[df.index[-1], 'Bull_Payout'] = bull_payout
        df.loc[df.index[-1], 'Bear_Payout'] = bear_payout

    # 4. Calculate RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.fillna(50)

    print(f'>>> Market data ready: {df.shape[0]} rows, {list(df.columns)}')

    return df, contract_info


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


def run_prediction_pipeline(limit=500, epochs=10, batch_size=32, 
                           rpc_url='https://bsc-dataseed1.binance.org/',
                           timeframe='1m'):
    """
    Run the full prediction pipeline: fetch real-time data, train model, make prediction.

    This is the end-to-end CRISP-DM workflow for the PancakeSwap Prediction Bot
    using only real-time data from PancakeSwap DEX and Prediction contract.

    Args:
        limit: Number of candles to fetch from PancakeSwap (default: 500).
        epochs: Number of training epochs (default: 10).
        batch_size: Training batch size (default: 32).
        rpc_url: BSC RPC endpoint URL.
        timeframe: Candle interval (default: '1m').

    Returns:
        Tuple of (model, history, trade_result) where trade_result is
        a dict with keys: action, confidence, ev_bull, ev_bear.
    """
    print(">>> FETCHING REAL-TIME DATA FROM PANCAKESWAP...")
    market_data, contract_info = fetch_live_market_data(
        timeframe=timeframe, 
        limit=limit, 
        rpc_url=rpc_url,
        use_contract=True
    )

    print(">>> PREPARING SEQUENCES...")
    X, y, scaler = create_sequences(market_data)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training Shapes: X={X_train.shape}, y={y_train.shape}")

    # Build and train model
    print(">>> TRAINING MODEL...")
    model = build_pancake_model()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Make prediction for current round
    print("\n>>> CURRENT ROUND PREDICTION ---")
    latest_data = X_test[-1]
    
    # Get current payouts from contract
    if contract_info is not None:
        current_bull_payout = contract_info['bull_payout']
        current_bear_payout = contract_info['bear_payout']
    else:
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
