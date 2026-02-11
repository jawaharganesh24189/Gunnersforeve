# Quick Start Guide

Get up and running with Arsenal FC match data and AI simulation in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/jawaharganesh24189/Gunnersforeve.git
cd Gunnersforeve

# Install dependencies
pip install -r requirements.txt
```

## Three Ways to Get Started

### 🌟 Option 1: Jupyter Notebook (Recommended - Easiest!)

**Everything in one place - no coding required!**

```bash
jupyter notebook arsenal_complete_notebook.ipynb
```

The notebook includes:
- ✅ Data collection from APIs
- ✅ AI match simulation (basic & tactical)
- ✅ Data analysis with pandas
- ✅ Beautiful visualizations
- ✅ Export to CSV/JSON

**Just run the cells from top to bottom!**

---

### ⚡ Option 2: Command Line (Quick & Powerful)

#### Simulate Matches

```bash
# Basic simulation - fast, great for large datasets
python scripts/simulate_matches.py --matches 38 --show-results --seed 42

# Tactical simulation - detailed, realistic match dynamics
python scripts/simulate_tactical.py Arsenal Liverpool --detailed --show-events
```

#### Fetch Real Data

```bash
# Fetch real match data (uses mock data without API key)
python scripts/fetch_data.py --season 2023 --format both
```

---

### 🐍 Option 3: Python API (Most Flexible)

```python
# Basic simulation (fast, for large datasets)
from src.simulator import create_simulated_dataset

dataset = create_simulated_dataset(
    num_matches=100,
    team="Arsenal",
    seed=42
)
dataset.to_csv("my_dataset.csv")
```

```python
# Tactical simulation (detailed, realistic)
from src.tactical_simulator import TacticalMatchSimulator

simulator = TacticalMatchSimulator(seed=42)

match_data, match_state = simulator.simulate_tactical_match(
    home_team="Arsenal",
    away_team="Manchester City",
    date="2024-03-31",
    detailed_events=True
)

# View match summary
print(simulator.get_match_summary())

# Access events
for event in match_state.events:
    if event.event_type.value == "goal":
        print(f"{event.minute}' - ⚽ GOAL by {event.team}!")
```

---

## Common Use Cases

### Generate Training Data for ML

```bash
# Generate 1000 matches quickly
python scripts/simulate_matches.py --matches 1000 --format csv --output-dir data/training
```

### Simulate a Specific Match

```bash
# Arsenal vs Manchester City with full tactical details
python scripts/simulate_tactical.py Arsenal "Manchester City" --detailed --show-events --save
```

### Analyze Team Performance

Open `arsenal_complete_notebook.ipynb` and run all cells to get:
- Match results breakdown
- Goals analysis
- Home vs Away performance
- xG analysis
- Form guide visualization

### Compare Playing Styles

```python
from src.tactical_simulator import TacticalMatchSimulator, ADVANCED_PL_TEAMS

# Check team profiles
arsenal = ADVANCED_PL_TEAMS["Arsenal"]
print(f"{arsenal.name}: {arsenal.preferred_formation.value}, {arsenal.preferred_style.value}")

# Simulate 100 matches to see results
simulator = TacticalMatchSimulator(seed=42)
# ... (see TACTICAL_GUIDE.md for full example)
```

---

## Next Steps

### Learn More
- 📖 [README.md](README.md) - Full project overview
- 📊 [DATA_SOURCES.md](DATA_SOURCES.md) - 8+ data sources explained
- 🎮 [SIMULATION.md](SIMULATION.md) - Basic simulation guide
- ⚽ [TACTICAL_GUIDE.md](TACTICAL_GUIDE.md) - Advanced tactical simulation
- 💻 [USAGE.md](USAGE.md) - Detailed usage instructions

### Example Datasets
Check `data/examples/` for sample data:
- `arsenal_matches_sample.json`
- `arsenal_matches_sample.csv`

### Get an API Key (Optional)
For real data: https://www.football-data.org/client/register

```bash
# Set up API key
cp .env.example .env
# Edit .env and add: FOOTBALL_DATA_API_KEY=your_key_here
```

---

## Troubleshooting

### "Module not found" Error
```bash
# Make sure you're in the project root
cd /path/to/Gunnersforeve

# Make sure dependencies are installed
pip install -r requirements.txt
```

### Jupyter Notebook Won't Open
```bash
# Install Jupyter if needed
pip install jupyter

# Try launching from project root
jupyter notebook arsenal_complete_notebook.ipynb
```

### Import Errors in Scripts
```bash
# Always run scripts from project root
cd /path/to/Gunnersforeve
python scripts/simulate_matches.py --matches 10
```

---

## Quick Commands Cheat Sheet

```bash
# Jupyter notebook (all-in-one)
jupyter notebook arsenal_complete_notebook.ipynb

# Basic AI simulation
python scripts/simulate_matches.py --matches 38 --seed 42

# Tactical simulation with events
python scripts/simulate_tactical.py Arsenal Liverpool --detailed --show-events

# Fetch real data
python scripts/fetch_data.py --season 2023

# Generate large dataset
python scripts/simulate_matches.py --matches 1000 --format csv

# Simulate specific matchup
python scripts/simulate_tactical.py Arsenal "Manchester City" --save

# Quick tactical simulation (no events)
python scripts/simulate_tactical.py Arsenal Chelsea --quick --matches 10
```

---

## Support

- 📝 Check documentation in repository
- 💡 See example datasets in `data/examples/`
- 🐛 Open issues on GitHub
- 📖 Read TACTICAL_GUIDE.md for advanced features

---

**Ready to start? Open the Jupyter notebook and run all cells!**

```bash
jupyter notebook arsenal_complete_notebook.ipynb
```

🎉 **Happy simulating!** ⚽
