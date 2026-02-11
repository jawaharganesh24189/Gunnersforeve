# Gunnersforeve
Dedicated for the gunners - Arsenal FC Match Data and Analysis

## Overview

This repository provides tools and datasets for collecting, processing, and analyzing Arsenal FC match data. It includes scripts for fetching real data from various sources, **AI-powered match simulation**, standardized data schemas, and example datasets for machine learning and statistical analysis.

## Features

- 🔍 **Multiple Data Sources**: Integration with football-data.org API and support for other sources
- 🤖 **AI Match Simulation**: Intelligent simulation of Arsenal and Premier League matches using statistical models
- ⚽ **Advanced Tactical Simulator**: Event-level simulation with formations, playing styles, and match dynamics
- 📊 **Standardized Schema**: Consistent data structure across all sources
- 🎯 **Ready for ML**: Structured datasets perfect for training machine learning models
- 📈 **Comprehensive Statistics**: Match results, team stats, player data, and advanced metrics
- 💾 **Multiple Formats**: Export to JSON, CSV, or other formats

## Quick Start

### 🌟 NEW: Self-Contained ML Notebook (Recommended for Learning!)

**For a complete standalone experience with NO external dependencies:**

```bash
jupyter notebook arsenal_ml_notebook_standalone.ipynb
```

This notebook has:
- ✅ **Zero external imports** - All code embedded directly
- ✅ **Complete ML pipeline** - From simulation to evaluation
- ✅ **Rich visualizations** - 5 comprehensive plots
- ✅ **Detailed explanations** - Markdown between every code cell
- ✅ **~500 lines of code** - Fully self-contained

Perfect for learning machine learning with football data!

---

### One-Click Access: Full-Featured Jupyter Notebook 📓

**For the complete system with all features:**

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook arsenal_complete_notebook.ipynb
```

The notebook includes:
- ✅ Real data collection
- ✅ AI match simulation  
- ✅ Data analysis & statistics
- ✅ Beautiful visualizations
- ✅ Export functionality

**Everything runs in your browser with no additional setup needed!**

---

### 📋 Which Notebook Should I Use?

| Feature | `arsenal_ml_notebook_standalone.ipynb` | `arsenal_complete_notebook.ipynb` |
|---------|----------------------------------------|-----------------------------------|
| **External Files** | ❌ None (fully self-contained) | ✅ Imports from src/ directory |
| **ML Models** | ✅ 2 models (Classification + Regression) | ❌ No ML models |
| **Visualizations** | ✅ 5 plots with detailed analysis | ✅ Multiple analysis plots |
| **Explanations** | ✅✅ Detailed markdown between all cells | ✅ Good documentation |
| **Data Collection** | ❌ Simulated data only | ✅ Real API integration |
| **Tactical Simulator** | ❌ Basic Poisson model | ✅ Advanced event-level |
| **Best For** | 📚 Learning ML & Analytics | 🚀 Production & Full Features |

**Choose:**
- 📚 **Learning/Education**: Use `arsenal_ml_notebook_standalone.ipynb` (completely self-contained!)
- 🚀 **Production/Analysis**: Use `arsenal_complete_notebook.ipynb` (full feature set)

---

### Alternative: Command Line Usage

#### Installation

```bash
# Clone the repository
git clone https://github.com/jawaharganesh24189/Gunnersforeve.git
cd Gunnersforeve

# Install dependencies
pip install -r requirements.txt
```

### Setting up API Keys (Optional)

To fetch live data, you'll need an API key from [football-data.org](https://www.football-data.org/client/register):

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# FOOTBALL_DATA_API_KEY=your_api_key_here
```

### Fetch Real Match Data

```bash
# Fetch data for the 2023-24 season (uses mock data if no API key)
python scripts/fetch_data.py --season 2023

# With API key
python scripts/fetch_data.py --api-key YOUR_KEY --season 2023 --format both
```

### Simulate Matches with AI

```bash
# Basic AI simulation - 20 Arsenal matches
python scripts/simulate_matches.py --matches 20 --show-results

# Advanced tactical simulation with event-level dynamics
python scripts/simulate_tactical.py Arsenal "Manchester City" --detailed --show-events

# Quick tactical simulation for multiple matches
python scripts/simulate_tactical.py Arsenal Chelsea --quick --matches 10

# Simulate a full Premier League round
python scripts/simulate_matches.py --league-round 2024-01-20

# Reproducible simulations with seed
python scripts/simulate_matches.py --matches 38 --seed 42 --team Arsenal
```

### Use Example Data

Example datasets are provided in `data/examples/`:
- `arsenal_matches_sample.json` - Match data in JSON format
- `arsenal_matches_sample.csv` - Match data in CSV format

## Project Structure

```
Gunnersforeve/
├── arsenal_complete_notebook.ipynb  # 🌟 ALL-IN-ONE NOTEBOOK - Start here!
├── data/
│   ├── raw/              # Raw data from APIs (gitignored)
│   ├── processed/        # Processed datasets (gitignored)
│   └── examples/         # Example datasets for reference
├── src/
│   ├── data_schema.py    # Data models and schemas
│   ├── data_collector.py # Data collection functions
│   └── simulator.py      # AI match simulation engine
├── scripts/
│   ├── fetch_data.py     # Script to fetch real data
│   └── simulate_matches.py # Script to simulate matches with AI
├── DATA_SOURCES.md       # Comprehensive list of data sources
├── SIMULATION.md         # AI simulation guide
├── USAGE.md             # Detailed usage guide
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## AI Match Simulation

The repository includes **two powerful simulation engines**:

### 1. Basic AI Simulator (`simulator.py`)
Fast statistical simulation using Poisson distribution:
- **Quick generation** of large datasets
- Realistic scores and statistics
- Team strength-based modeling
- Perfect for ML training data

### 2. Advanced Tactical Simulator (`tactical_simulator.py`)
Sophisticated event-level simulation with tactical depth:
- **Minute-by-minute** match simulation
- **Formations**: 4-3-3, 4-4-2, 3-5-2, 4-2-3-1, etc.
- **Playing styles**: Possession, Counter-attack, High Press, Defensive, etc.
- **Match dynamics**: Momentum, energy, morale
- **Event tracking**: Goals, shots, fouls, cards, corners with timestamps
- **Tactical adjustments**: Half-time changes based on score
- **Detailed team profiles**: 7+ tactical attributes per team

### Tactical Simulation Features

### Tactical Simulation Features
- **Realistic Scores**: Uses Poisson distribution based on team strengths
- **Team Profiles**: 20 Premier League teams with attack, defense, midfield ratings
- **Tactical Attributes**: Pressing ability, passing quality, pace, physicality, creativity, discipline
- **Formations & Styles**: Teams play with their preferred formations and tactics
- **Match Events**: Minute-by-minute tracking of shots, goals, fouls, cards, corners
- **Match Dynamics**: Momentum shifts, energy levels, morale changes
- **Tactical Adjustments**: Teams adjust tactics at half-time based on score
- **Advanced Stats**: Generates possession, shots, xG, corners, fouls, cards
- **Home Advantage**: Accounts for home field advantage
- **Form Factor**: Considers recent team performance
- **Reproducible**: Use seeds for consistent results

### Example Usage - Tactical Simulation

```python
from src.tactical_simulator import TacticalMatchSimulator

# Create tactical simulator
simulator = TacticalMatchSimulator(seed=42)

# Simulate with detailed events
match_data, match_state = simulator.simulate_tactical_match(
    home_team="Arsenal",
    away_team="Manchester City",
    date="2024-03-31",
    detailed_events=True  # Enable minute-by-minute simulation
)

# Access detailed events
for event in match_state.events:
    if event.event_type.value == "goal":
        print(f"{event.minute}' - ⚽ GOAL by {event.team}!")

# Get match summary
print(simulator.get_match_summary())
```

### Example Usage - Basic Simulation

```python
from src.simulator import FootballMatchSimulator

# Create simulator
simulator = FootballMatchSimulator(seed=42)

# Simulate a match
match = simulator.simulate_match(
    home_team="Arsenal",
    away_team="Manchester City",
    date="2024-03-31"
)

print(f"{match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
print(f"xG: {match.home_stats.xg} - {match.away_stats.xg}")
```

## Data Sources

The repository supports multiple data sources for Arsenal match data:

- **football-data.org API** - Match results and statistics
- **Football-Data.co.uk** - Historical CSV data
- **FBref.com** - Advanced statistics and metrics
- **StatsBomb Open Data** - Event-level match data
- **Kaggle Datasets** - Various football datasets
- **AI Simulation** - Generate synthetic training data

See [DATA_SOURCES.md](DATA_SOURCES.md) for a complete list with details and usage guidelines.

## Data Schema

All match data follows a standardized schema defined in `src/data_schema.py`:

```python
- Match metadata (date, time, competition, venue)
- Team information and scores
- Detailed statistics (shots, possession, corners, etc.)
- Advanced metrics (xG, xA when available)
- Player-level data (lineups, goals, assists)
```

## Usage Examples

### Loading Data in Python

```python
import json
from src.data_schema import MatchData, Dataset

# Load JSON dataset
with open('data/examples/arsenal_matches_sample.json', 'r') as f:
    data = json.load(f)
    dataset = Dataset(**data)

# Access matches
for match in dataset.matches:
    print(f"{match.date}: {match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
```

### Loading Data with Pandas

```python
import pandas as pd

# Load CSV dataset
df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Basic analysis
print(f"Total matches: {len(df)}")
print(f"Average possession: {df['home_possession'].mean():.1f}%")
```

### Simulate Training Data for ML

```python
from src.simulator import create_simulated_dataset

# Create large training dataset
dataset = create_simulated_dataset(
    num_matches=1000,
    team="Arsenal",
    season="2023-24",
    seed=42
)

# Export for training
dataset.to_csv("data/training/arsenal_1000_matches.csv")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Data Usage Guidelines

- Always respect API rate limits
- Cache data locally to minimize requests
- Attribute data sources appropriately
- Use data for non-commercial, educational purposes
- Follow each source's terms of service

## License

This project is for educational and non-commercial use.

## Acknowledgments

- Arsenal FC - The Gunners
- football-data.org for providing match data APIs
- All open-source contributors to football data projects
