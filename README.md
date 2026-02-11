# Gunnersforeve
Dedicated for the gunners

## Football League Simulation with Tactical AI

This repository contains a comprehensive Jupyter Notebook that simulates a football league and trains a tactical AI using deep learning.

### Features

- **Player Class**: Generate players with random stats based on their position (GK, DEF, MID, FWD)
- **Team Class**: Create teams with 11 players in a realistic formation
- **League Class**: Manage multiple teams and simulate full seasons
- **Match Simulation**: Generate realistic match results and tracking data (x,y coordinates)
- **Tactical AI Model**: BiLSTM + Multi-Head Attention neural network to predict goal probability
- **Match Functions**: Simulate random matches or specific matchups
- **Visualization**: Display ideal attack patterns based on AI predictions

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Open the Jupyter Notebook:

```bash
jupyter notebook football_league_tactical_ai.ipynb
```

Run all cells to:
1. Create a football league with 6 teams
2. Simulate a complete season
3. Generate tracking data from matches
4. Train a deep learning model to predict goal probability
5. Visualize ideal attack patterns
6. Simulate new matches

### Model Architecture

The tactical AI uses:
- **BiLSTM layers**: Capture temporal movement patterns of players
- **Multi-Head Attention**: Model player interactions and tactical relationships
- **Goal Prediction**: Binary classification to predict goal probability from position sequences

### Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebook
