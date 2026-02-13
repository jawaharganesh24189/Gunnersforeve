# Gunnersforeve - Football Tactics Transformer

A Keras-based Transformer model that generates passing tactics from the backline to the opposite goal, considering different oppositions, formations, and tactical contexts.

**🆕 NEW: Enhanced notebook with multi-league support, player stats, and match history!** See [enhanced_tactics_transformer_notebook.ipynb](enhanced_tactics_transformer_notebook.ipynb)

**🆕 NEW: PancakeSwap Prediction Bot** — BiLSTM + Attention model for BNB 5-minute price prediction. See [pancake_predictor_notebook.ipynb](pancake_predictor_notebook.ipynb)

## Overview

This project implements a state-of-the-art transformer neural network architecture to generate intelligent passing sequences in football. The model can analyze tactical situations including:

- Team formations (4-3-3, 4-4-2, 3-5-2, etc.)
- Opposition formations
- Player positions on the field
- Ball position
- Tactical context (counter-attack, possession, build from back, etc.)

And generate optimal passing sequences from defense to attack.

## Features

### Core Features
- **Multi-Head Attention Mechanism**: Captures complex relationships between players and positions
- **Positional Encoding**: Understands sequence order and field positions
- **Encoder-Decoder Architecture**: Processes tactical input and generates passing sequences
- **Customizable**: Easily adjust model parameters, formations, and tactical contexts
- **Extensible**: Built with modularity in mind for easy integration and extension

### Enhanced Features (NEW)
- **🌍 Multi-League Support**: 40+ teams from Premier League, Serie A, Ligue 1, La Liga, and Bundesliga
- **👤 Player Statistics**: Individual player ratings with position-specific calculations
- **📊 Team Attributes**: Attack/defense ratings, possession style, pressing intensity
- **📈 Match History**: Training on actual match data with outcomes and statistics

## Project Structure

```
Gunnersforeve/
├── src/
│   ├── transformer_model.py      # Core transformer architecture
│   ├── data_preprocessing.py     # Data encoding and dataset creation
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference and tactics generation
│   ├── teams_data.py             # Multi-league teams database (NEW)
│   ├── player_stats.py           # Player statistics system (NEW)
│   ├── match_history.py          # Match data and outcomes (NEW)
│   └── pancake_predictor.py      # PancakeSwap BNB prediction bot (NEW)
├── examples/
│   └── usage_examples.py         # Example usage demonstrations
├── enhanced_tactics_transformer_notebook.ipynb  # Standalone enhanced notebook (NEW)
├── pancake_predictor_notebook.ipynb             # PancakeSwap prediction notebook (NEW)
├── arsenal_ml_notebook_standalone.ipynb         # Original standalone notebook
├── tests/                        # Test files (to be added)
├── requirements.txt              # Python dependencies
├── ENHANCED_NOTEBOOK_README.md   # Enhanced notebook documentation (NEW)
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

### Option 1: Enhanced Standalone Notebook (Recommended)

Use the new enhanced notebook with multi-league support:

```bash
jupyter notebook enhanced_tactics_transformer_notebook.ipynb
```

This notebook includes:
- 40+ teams from 5 major leagues
- 25+ players with detailed statistics
- Sample match data from multiple leagues
- All code embedded (no external files needed)

See [ENHANCED_NOTEBOOK_README.md](ENHANCED_NOTEBOOK_README.md) for details.

### Option 2: PancakeSwap Prediction Bot

Run the BNB 5-minute prediction model notebook:

```bash
jupyter notebook pancake_predictor_notebook.ipynb
```

Or use the Python module directly:

```python
from src.pancake_predictor import (
    generate_market_data,
    create_sequences,
    build_pancake_model,
    trade_logic,
    run_prediction_pipeline
)

# Run the full pipeline (generate data, train, predict)
model, history, result = run_prediction_pipeline()

# Or step by step:
market_data = generate_market_data(n_minutes=10000)
X, y, scaler = create_sequences(market_data)
model = build_pancake_model()
decision, prob, ev_bull, ev_bear = trade_logic(
    model, X[-1], bull_payout=1.95, bear_payout=1.95
)
```

### Option 3: Python Modules

#### 1. Explore Teams and Players

```python
from src import get_team_by_name, get_player_by_name, get_teams_by_league, League

# Get teams from different leagues
arsenal = get_team_by_name("Arsenal")
napoli = get_team_by_name("Napoli")
psg = get_team_by_name("Paris Saint-Germain")

# Get all Serie A teams
serie_a_teams = get_teams_by_league(League.SERIE_A)

# Get player stats
saliba = get_player_by_name("Saliba")
print(f"{saliba.name}: Overall {saliba.overall}, CB rating: {saliba.get_position_rating('CB')}")
```

#### 2. Load Match History

```python
from src import load_match_history

loader = load_match_history()
stats = loader.get_statistics()
print(f"Total matches: {stats['total_matches']}")
```

#### 3. Run Example Scripts

See the model in action with example scenarios:

```bash
python examples/usage_examples.py
```

#### 4. Train the Model

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

# Generate tactics
tactics = generator.generate_tactics(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=player_positions,
    temperature=0.8
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

The transformer model consists of:

1. **Embedding Layers**: Convert tactical tokens to dense vectors
2. **Positional Encoding**: Add position information to embeddings
3. **Encoder Stack**: 
   - Multi-head self-attention
   - Position-wise feed-forward networks
   - Layer normalization and residual connections
4. **Decoder Stack**:
   - Masked multi-head self-attention
   - Encoder-decoder attention
   - Position-wise feed-forward networks
5. **Output Layer**: Projects to vocabulary of passing actions

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
