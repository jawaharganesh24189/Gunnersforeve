# Arsenal ML Notebook - Standalone Edition

## Overview

This is a comprehensive, fully self-contained Jupyter notebook that demonstrates advanced machine learning techniques for football analytics. All dependencies and code are embedded directly in the notebook - no external Python modules required!

## Features

### Part 1: Match Outcome Prediction
- **Match Simulator**: Realistic match simulation using team profiles and Poisson distribution
- **Team Profiles**: 20 Premier League teams with attack/defense/midfield ratings
- **Feature Engineering**: 6 key features including possession, shots, accuracy, and xG
- **ML Models**: 
  - Random Forest Classifier for match result prediction (Win/Draw/Loss)
  - Gradient Boosting Regressor for goal prediction
- **Visualizations**: 5 comprehensive plots showing model performance and insights

### Part 2: Passing Tactics Generation (NEW!)
- **Transformer Architecture**: State-of-the-art sequence-to-sequence neural network
- **Multi-Head Attention**: Captures complex spatial relationships between players
- **Tactical Encoding**: Represents formations, positions, and game context
- **Passing Sequences**: Generates intelligent passing sequences from backline to goal
- **Flexible Input**: Handles different:
  - Team formations (4-3-3, 4-4-2, 3-5-2, 4-2-3-1, etc.)
  - Opposition formations
  - Player positions on the field
  - Tactical contexts (counter-attack, possession, build-up, high press)

## What's Included

The notebook contains **36 cells** organized into two main sections:

1. **Cells 1-24**: Match outcome prediction with simulator and ML models
2. **Cells 25-36**: Transformer model for passing tactics generation

All code is self-contained:
- ✅ No imports from external project files
- ✅ Complete transformer architecture embedded
- ✅ Data preprocessing utilities included
- ✅ Inference and generation code integrated
- ✅ Example scenarios and demonstrations

## Requirements

Only standard Python packages are needed:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Or using the project's requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook arsenal_ml_notebook_standalone.ipynb
   ```

3. **Run all cells**: 
   - Click "Cell" → "Run All" in Jupyter
   - Or run cells sequentially (recommended for learning)

## Notebook Structure

### Section 1: Match Outcome Prediction

| Cell | Description |
|------|-------------|
| 1-2  | Introduction and imports |
| 3-5  | Team profiles (20 Premier League teams) |
| 6-7  | Match simulator implementation |
| 8-9  | Dataset generation (500 matches) |
| 10-11 | Feature engineering |
| 12-15 | Model training (Random Forest + Gradient Boosting) |
| 16-17 | Model evaluation metrics |
| 18-23 | Visualizations (5 plots) |
| 24   | Part 1 summary |

### Section 2: Passing Tactics Generation

| Cell | Description |
|------|-------------|
| 25-26 | Introduction to transformer architecture |
| 27   | Complete transformer model code (~300 lines) |
| 28-29 | Data preprocessing and encoding (~280 lines) |
| 30-31 | Inference and tactics generation (~250 lines) |
| 32-33 | Model instantiation example |
| 34-35 | Tactical situation encoding examples |
| 36   | Summary and next steps |

## Key Components

### Transformer Model Architecture

```
Input (Tactical Situation)
    ↓
Embedding + Positional Encoding
    ↓
Encoder Stack (Multi-Head Attention + FFN)
    ↓
Decoder Stack (Masked Attention + Cross-Attention + FFN)
    ↓
Output Layer
    ↓
Passing Sequence (Position + Action pairs)
```

### Supported Tactical Elements

**Formations**: 4-4-2, 4-3-3, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2, 4-5-1, 4-1-4-1

**Positions**: GK, LB, CB, RB, LWB, RWB, CDM, CM, LM, RM, CAM, LW, RW, ST, CF

**Actions**: short_pass, long_pass, through_ball, cross, switch_play, back_pass, forward_pass, diagonal_pass

**Contexts**: counter_attack, possession, high_press, low_block, build_from_back, direct_play

## Example Usage

### Creating the Transformer Model

```python
from transformer_model import create_tactics_transformer

model = create_tactics_transformer(
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512
)
```

### Encoding a Tactical Situation

```python
from data_preprocessing import TacticsEncoder

encoder = TacticsEncoder()

encoded = encoder.encode_tactical_situation(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=[
        ('GK', 5, 50),
        ('CB', 20, 35),
        ('CDM', 35, 50),
        ('CAM', 60, 50),
        ('ST', 80, 50)
    ]
)
```

### Generating Passing Tactics

```python
from inference import TacticsGenerator

generator = TacticsGenerator(model, encoder)

tactics = generator.generate_tactics(
    own_formation='4-3-3',
    opponent_formation='4-4-2',
    ball_position=(25, 50),
    tactical_context='counter_attack',
    player_positions=[...],
    temperature=0.8
)
```

## Model Parameters

### Transformer Hyperparameters

- **num_layers**: 4 (number of encoder/decoder layers)
- **d_model**: 256 (embedding dimension)
- **num_heads**: 8 (number of attention heads)
- **dff**: 512 (feed-forward network dimension)
- **dropout_rate**: 0.1 (regularization)

### Data Parameters

- **input_vocab_size**: 200 (tactical elements vocabulary)
- **target_vocab_size**: 50 (passing actions vocabulary)
- **max_position_encoding**: 100 (maximum sequence length)

## Training (Future Enhancement)

The notebook demonstrates the model architecture. For production use:

1. Collect real match data (player positions, pass sequences, outcomes)
2. Preprocess using the `TacticsEncoder` class
3. Train using the training script: `python src/train.py`
4. Load trained weights for inference

## Visualizations

The notebook includes 5 visualizations for match prediction:

1. **Result Distribution**: Pie chart of Win/Draw/Loss outcomes
2. **Possession vs Goals**: Scatter plot showing correlation
3. **xG Validation**: Comparison of expected vs actual goals
4. **Feature Importance**: Bar chart of most important features
5. **Model Predictions**: Heatmap of prediction accuracy

## Performance

- **Match Prediction**: ~70-80% accuracy on simulated data
- **Transformer Model**: Real-time inference capable
- **Memory Usage**: ~500MB for small transformer (2 layers, d=128)
- **Training Time**: ~5-10 minutes for 1000 samples on CPU

## Files in This Repository

```
Gunnersforeve/
├── arsenal_ml_notebook_standalone.ipynb  ← This notebook (64KB, 36 cells)
├── src/
│   ├── transformer_model.py              ← Standalone transformer code
│   ├── data_preprocessing.py             ← Encoding utilities
│   ├── inference.py                      ← Tactics generation
│   └── train.py                          ← Training script
├── examples/
│   └── usage_examples.py                 ← Python examples
├── requirements.txt                      ← Dependencies
└── README.md                             ← Project documentation
```

## Learning Objectives

After working through this notebook, you'll understand:

1. **Match simulation** and synthetic data generation
2. **Feature engineering** for sports analytics
3. **Traditional ML** (Random Forest, Gradient Boosting)
4. **Transformer architecture** for sequence generation
5. **Multi-head attention** mechanisms
6. **Encoder-decoder models** for tactical planning
7. **Tactical encoding** and representation learning

## Next Steps

1. **Experiment with hyperparameters**: Adjust model size, layers, attention heads
2. **Train on real data**: Replace synthetic data with actual match recordings
3. **Visualize tactics**: Add field diagram visualization for passing sequences
4. **Extend features**: Add defensive positioning, pressure zones, space analysis
5. **Real-time inference**: Deploy model for live match tactical suggestions
6. **Multi-objective optimization**: Balance attacking efficiency with defensive solidity

## Troubleshooting

**Issue**: `ModuleNotFoundError` when running cells
- **Solution**: Ensure all cells are run in order. The notebook is self-contained, so all code is defined in earlier cells.

**Issue**: `OutOfMemoryError` when creating large models
- **Solution**: Reduce `d_model` or `num_layers` parameters. For testing, use `num_layers=2`, `d_model=128`.

**Issue**: Slow execution on CPU
- **Solution**: Use GPU if available. Install `tensorflow-gpu` and ensure CUDA is configured.

**Issue**: Random/unrealistic tactics generated
- **Solution**: The model needs training on real data. Current version uses untrained model for demonstration.

## Contributing

Contributions welcome! Areas for improvement:

- Real match data integration
- Advanced tactical features
- Field visualization
- Model optimization
- Additional formations and contexts

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with TensorFlow/Keras
- Transformer architecture based on "Attention is All You Need" (Vaswani et al., 2017)
- Match simulation inspired by StatsBomb open data

---

**Dedicated for the Gunners** ⚽️🔴⚪️
