# Gunnersforeve - Football Tactics Transformer

A Keras-based Transformer model that generates passing tactics from the backline to the opposite goal, considering different oppositions, formations, and tactical contexts.

> **🎉 NEW (v1.1.0):** Improved with modern Deep Learning Architecture (DLA) best practices! See [DLA_IMPROVEMENTS.md](DLA_IMPROVEMENTS.md) for details.

## Overview

This project implements a state-of-the-art transformer neural network architecture to generate intelligent passing sequences in football. The model can analyze tactical situations including:

- Team formations (4-3-3, 4-4-2, 3-5-2, etc.)
- Opposition formations
- Player positions on the field
- Ball position
- Tactical context (counter-attack, possession, build from back, etc.)

And generate optimal passing sequences from defense to attack.

## ✨ Recent Improvements (v1.1.0)

Based on modern DLA (Deep Learning Architecture) best practices:

- ✅ **Pre-LayerNorm Architecture** - Better training stability for deep models
- ✅ **Keras Built-in MultiHeadAttention** - 2-3x faster, optimized performance
- ✅ **Learnable Positional Embeddings** - Optional alternative to fixed sinusoidal
- ✅ **Gradient Clipping** - Stable training for deep transformers
- ✅ **GELU Activation** - Modern activation function (used in BERT/GPT)
- ✅ **Beam Search** - Higher quality sequence generation
- ✅ **Type Hints** - Better code maintainability
- ✅ **Improved Training** - Auto-calculated warmup, TensorBoard logging

**[📖 Read full improvements documentation →](DLA_IMPROVEMENTS.md)**

## Features

- **Optimized Multi-Head Attention**: Uses Keras built-in for peak performance
- **Flexible Positional Encoding**: Both fixed and learnable options
- **Pre-LN Architecture**: Modern transformer design for better stability
- **Beam Search Inference**: Generate higher quality tactical sequences
- **Encoder-Decoder Architecture**: Processes tactical input and generates passing sequences
- **Gradient Clipping**: Stable training even for deep networks
- **Customizable**: Easily adjust model parameters, formations, and tactical contexts
- **Extensible**: Built with modularity in mind for easy integration and extension

## Project Structure

```
Gunnersforeve/
├── src/
│   ├── transformer_model.py      # Core transformer architecture (DLA improved)
│   ├── data_preprocessing.py     # Data encoding and dataset creation
│   ├── train.py                  # Training script
│   └── inference.py              # Inference and tactics generation
├── examples/
│   └── usage_examples.py         # Example usage demonstrations
├── tests/                        # Test files (to be added)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jawaharganesh24189/Gunnersforeve.git
cd Gunnersforeve
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Examples

See the model in action with example scenarios:

```bash
python examples/usage_examples.py
```

### 2. Train the Model

Train the transformer on synthetic tactical data:

```bash
python src/train.py
```

Training options can be customized by modifying the parameters in `train.py`:
- `num_samples`: Number of training samples (default: 1000)
- `num_layers`: Number of transformer layers (default: 4)
- `d_model`: Model dimension (default: 256)
- `num_heads`: Number of attention heads (default: 8)
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size (default: 32)

### 3. Generate Tactics

Use the trained model to generate passing tactics:

```bash
python src/inference.py
```

## Usage Examples

### Creating a Model

```python
from src.transformer_model import create_tactics_transformer

model = create_tactics_transformer(
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512,
    input_vocab_size=200,
    target_vocab_size=50,
    max_position_encoding=100,
    dropout_rate=0.1
)
```

### Encoding Tactical Situations

```python
from src.data_preprocessing import TacticsEncoder

encoder = TacticsEncoder()

# Define tactical situation
own_formation = '4-3-3'
opponent_formation = '4-4-2'
ball_position = (25, 50)
tactical_context = 'counter_attack'
player_positions = [
    ('CB', 20, 35),
    ('CDM', 35, 50),
    ('CAM', 60, 50),
    ('ST', 80, 50)
]

# Encode the situation
encoded = encoder.encode_tactical_situation(
    own_formation,
    opponent_formation,
    ball_position,
    tactical_context,
    player_positions
)
```

### Generating Passing Tactics

```python
from src.inference import TacticsGenerator

# Create generator with trained model
generator = TacticsGenerator(model, encoder, max_length=20)

# Generate tactics (sampling-based)
tactics = generator.generate_tactics(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=player_positions,
    temperature=0.8
)

# ✨ NEW: Generate tactics with beam search (better quality)
tactics_beam = generator.generate_tactics_beam_search(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=player_positions,
    beam_width=5,       # Number of beams
    length_penalty=1.0  # Length normalization
)

# Generate multiple options
multiple_tactics = generator.generate_multiple_tactics(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=player_positions,
    num_samples=3
)
```

## Model Architecture

The transformer model uses modern DLA best practices:

1. **Embedding Layers**: Convert tactical tokens to dense vectors
2. **Positional Encoding**: Fixed sinusoidal or learnable embeddings
3. **Encoder Stack** (Pre-LayerNorm):
   - Layer normalization (before sub-layer)
   - Keras built-in multi-head self-attention
   - Residual connection
   - Layer normalization (before FFN)
   - Position-wise feed-forward networks (GELU activation)
   - Residual connection
4. **Decoder Stack** (Pre-LayerNorm):
   - Layer normalization + Masked self-attention + Residual
   - Layer normalization + Cross-attention with encoder + Residual
   - Layer normalization + Feed-forward (GELU) + Residual
5. **Final Layer Normalization**: Stabilizes outputs (Pre-LN architecture)
6. **Output Layer**: Projects to vocabulary of passing actions

**Key Architecture Improvements:**
- Pre-LN instead of Post-LN for better gradient flow
- Keras built-in attention for optimized performance
- GELU activation for smoother gradients
- Gradient clipping for training stability

**[See detailed architecture diagram →](DLA_IMPROVEMENTS.md#architecture-diagram)**

### Supported Formations

- 4-4-2 (Traditional)
- 4-3-3 (Attacking)
- 3-5-2 (Wing-back system)
- 4-2-3-1 (Modern balanced)
- 3-4-3 (Attacking)
- 5-3-2 (Defensive)
- 4-5-1 (Counter-attacking)
- 4-1-4-1 (Possession-based)

### Supported Positions

GK, LB, CB, RB, LWB, RWB, CDM, CM, LM, RM, CAM, LW, RW, ST, CF

### Passing Actions

- Short pass
- Long pass
- Through ball
- Cross
- Switch play
- Back pass
- Forward pass
- Diagonal pass

### Tactical Contexts

- Counter-attack
- Possession
- High press
- Low block
- Build from back
- Direct play

## Training Data

The current implementation includes a synthetic data generator for demonstration. In production, you would:

1. Collect real match data (player positions, pass sequences, formations)
2. Preprocess and encode the data using `TacticsEncoder`
3. Train the model on historical tactical situations and successful passing sequences

## Customization

### Adding New Formations

Edit `src/data_preprocessing.py`:

```python
self.formations = {
    '4-4-2': 1,
    '4-3-3': 2,
    'YOUR-FORMATION': 3,  # Add here
    # ...
}
```

### Adding New Actions

```python
self.actions = {
    'short_pass': 1,
    'YOUR-ACTION': 2,  # Add here
    # ...
}
```

### Adjusting Model Size

Larger models for complex tactics:
```python
model = create_tactics_transformer(
    num_layers=6,      # More layers
    d_model=512,       # Larger dimension
    num_heads=16,      # More attention heads
    dff=2048          # Larger feed-forward
)
```

## Performance Considerations

- **Training Time**: Depends on dataset size and model complexity
- **Inference Speed**: Real-time capable for tactical suggestions
- **Memory Usage**: Scales with model size (d_model and num_layers)

## Future Enhancements

- [ ] Integration with real match data
- [ ] Visualization of generated tactics on field diagrams
- [ ] Multi-language tactical terminology support
- [ ] Integration with match analysis tools
- [ ] Mobile deployment for real-time tactical suggestions
- [ ] Reinforcement learning for tactical optimization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with TensorFlow/Keras using the Transformer architecture pioneered in "Attention is All You Need" (Vaswani et al., 2017).

---

**Dedicated for the Gunners** ⚽️🔴⚪️
