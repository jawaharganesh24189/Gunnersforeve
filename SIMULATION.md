# AI Match Simulation Guide

This guide explains how to use the AI-powered football match simulation engine to generate training datasets for Arsenal FC and Premier League matches.

## Overview

The AI simulation engine (`src/simulator.py`) uses statistical models to generate realistic football match outcomes. It's perfect for:

- Creating large training datasets for machine learning
- Testing prediction algorithms
- Analyzing "what-if" scenarios
- Generating synthetic data when real data is unavailable

## How It Works

### Statistical Model

The simulator uses a **Poisson-based scoring model** that considers:

1. **Team Strength Ratings**
   - Attack strength (0-100)
   - Defense strength (0-100)
   - Midfield strength (0-100)

2. **Situational Factors**
   - Home advantage (typically 10-12 points)
   - Current form (0-10 scale)
   - Opposition quality

3. **Realistic Statistics**
   - Possession (correlated with midfield strength)
   - Shots and shots on target
   - Expected Goals (xG)
   - Corners, fouls, cards

### Team Profiles

The simulator includes profiles for all 20 Premier League teams (2023-24 season):

**Top Teams:**
- Manchester City: 92 Attack, 85 Defense, 90 Midfield
- Arsenal: 88 Attack, 82 Defense, 86 Midfield
- Liverpool: 90 Attack, 80 Defense, 87 Midfield

**Mid-Table:**
- Newcastle: 77 Attack, 80 Defense, 78 Midfield
- Brighton: 75 Attack, 73 Defense, 77 Midfield
- Aston Villa: 76 Attack, 74 Defense, 75 Midfield

**Lower Table:**
- Luton Town: 60 Attack, 63 Defense, 62 Midfield
- Burnley: 62 Attack, 65 Defense, 63 Midfield
- Sheffield United: 58 Attack, 68 Defense, 60 Midfield

## Basic Usage

### Command Line

#### Simulate Arsenal Season

```bash
# Simulate 38 matches (full season)
python scripts/simulate_matches.py --matches 38 --team Arsenal

# Show detailed results
python scripts/simulate_matches.py --matches 20 --show-results

# Reproducible results with seed
python scripts/simulate_matches.py --matches 38 --seed 42
```

#### Simulate League Round

```bash
# Simulate all 10 matches for a specific matchday
python scripts/simulate_matches.py --league-round 2024-01-20

# With specific seed for reproducibility
python scripts/simulate_matches.py --league-round 2024-02-03 --seed 100
```

#### Output Options

```bash
# JSON only
python scripts/simulate_matches.py --matches 20 --format json

# CSV only
python scripts/simulate_matches.py --matches 20 --format csv

# Both formats (default)
python scripts/simulate_matches.py --matches 20 --format both

# Custom output directory
python scripts/simulate_matches.py --matches 20 --output-dir my_simulations/
```

### Python API

#### Simulate Single Match

```python
from src.simulator import FootballMatchSimulator

# Create simulator
simulator = FootballMatchSimulator(seed=42)

# Simulate a match
match = simulator.simulate_match(
    home_team="Arsenal",
    away_team="Manchester City",
    date="2024-03-31",
    competition="Premier League",
    season="2023-24"
)

# Access match data
print(f"Score: {match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
print(f"Possession: {match.home_stats.possession}% - {match.away_stats.possession}%")
print(f"Shots: {match.home_stats.shots} - {match.away_stats.shots}")
print(f"xG: {match.home_stats.xg} - {match.away_stats.xg}")
print(f"Venue: {match.venue}")
print(f"Attendance: {match.attendance:,}")
```

#### Simulate Full Season

```python
from src.simulator import FootballMatchSimulator

simulator = FootballMatchSimulator(seed=42)

# Simulate 38 Arsenal matches
matches = simulator.simulate_season(
    team="Arsenal",
    season="2023-24",
    num_matches=38
)

# Analyze results
wins = sum(1 for m in matches if (
    (m.home_team == "Arsenal" and m.home_score > m.away_score) or
    (m.away_team == "Arsenal" and m.away_score > m.home_score)
))
draws = sum(1 for m in matches if m.home_score == m.away_score)
losses = len(matches) - wins - draws

print(f"Season Record: {wins}W {draws}D {losses}L")
print(f"Points: {wins * 3 + draws}")
```

#### Create Training Dataset

```python
from src.simulator import create_simulated_dataset

# Create large dataset for ML training
dataset = create_simulated_dataset(
    num_matches=1000,
    team="Arsenal",
    season="2023-24",
    seed=42
)

# Save for training
dataset.to_csv("data/training/arsenal_training_1000.csv")

print(f"Created {len(dataset.matches)} simulated matches")
```

#### Simulate League Round

```python
from src.simulator import FootballMatchSimulator

simulator = FootballMatchSimulator(seed=42)

# Simulate all 10 Premier League matches for one round
matches = simulator.simulate_league_round(
    date="2024-01-20",
    season="2023-24"
)

# Display results
for match in matches:
    print(f"{match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
```

## Advanced Usage

### Custom Team Profiles

You can create custom team profiles for simulation:

```python
from src.simulator import TeamProfile, FootballMatchSimulator

# Create custom team
custom_team = TeamProfile(
    name="Super Arsenal",
    attack_strength=95,
    defense_strength=90,
    midfield_strength=92,
    form=9.5,
    home_advantage=15
)

# Add to simulator's team dictionary
from src import simulator as sim_module
sim_module.PREMIER_LEAGUE_TEAMS["Super Arsenal"] = custom_team

# Now simulate with custom team
simulator = FootballMatchSimulator()
match = simulator.simulate_match(
    home_team="Super Arsenal",
    away_team="Manchester City",
    date="2024-04-01"
)
```

### Batch Simulations

Generate multiple datasets with different seeds:

```python
from src.simulator import create_simulated_dataset

datasets = []
for seed in range(10):
    dataset = create_simulated_dataset(
        num_matches=38,
        team="Arsenal",
        season="2023-24",
        seed=seed
    )
    datasets.append(dataset)
    
    # Save each
    dataset.to_csv(f"data/training/arsenal_sim_{seed}.csv")

print(f"Created {len(datasets)} different season simulations")
```

### Parameter Sensitivity Analysis

Test how different team strengths affect outcomes:

```python
from src.simulator import FootballMatchSimulator, PREMIER_LEAGUE_TEAMS

simulator = FootballMatchSimulator(seed=42)

# Vary Arsenal's attack strength
results = []
for attack in range(75, 96, 5):
    PREMIER_LEAGUE_TEAMS["Arsenal"].attack_strength = attack
    
    matches = simulator.simulate_season(team="Arsenal", num_matches=38)
    wins = sum(1 for m in matches if (
        (m.home_team == "Arsenal" and m.home_score > m.away_score) or
        (m.away_team == "Arsenal" and m.away_score > m.home_score)
    ))
    
    results.append((attack, wins))
    print(f"Attack {attack}: {wins} wins")
```

## Statistics and Metrics

### Understanding xG (Expected Goals)

The simulator generates realistic xG values:
- Based on shots on target
- Typical range: 0.10-0.18 per shot on target
- Correlated with final score but allows variance

### Possession Distribution

Possession is calculated based on:
- Midfield strength differential
- Typical range: 30-70%
- Mean: ~50% ± team strength difference

### Shot Generation

Shots are generated based on:
- Attack strength
- Actual goals scored (more goals → more shots)
- Random variation (±3 shots)
- Typical range: 5-25 shots per match

## Use Cases

### 1. Machine Learning Training Data

```python
# Create large balanced dataset
dataset = create_simulated_dataset(num_matches=5000, seed=42)
dataset.to_csv("data/ml/training_5000.csv")

# Use for prediction models
import pandas as pd
df = pd.read_csv("data/ml/training_5000.csv")

# Features: possession, shots, xG, etc.
# Target: win/draw/loss
```

### 2. Testing Prediction Algorithms

```python
# Simulate known scenarios to test algorithms
simulator = FootballMatchSimulator(seed=42)

test_cases = [
    ("Arsenal", "Manchester City"),
    ("Arsenal", "Luton Town"),
    ("Sheffield United", "Arsenal"),
]

for home, away in test_cases:
    match = simulator.simulate_match(home, away, "2024-01-01")
    # Test your prediction algorithm here
```

### 3. Season Projections

```python
# Simulate multiple seasons to get distribution
results = []
for seed in range(100):
    matches = FootballMatchSimulator(seed=seed).simulate_season(
        team="Arsenal",
        num_matches=38
    )
    
    points = sum(3 if (
        (m.home_team == "Arsenal" and m.home_score > m.away_score) or
        (m.away_team == "Arsenal" and m.away_score > m.home_score)
    ) else (1 if m.home_score == m.away_score else 0) for m in matches)
    
    results.append(points)

import numpy as np
print(f"Expected points: {np.mean(results):.1f} ± {np.std(results):.1f}")
print(f"Range: {min(results)} - {max(results)}")
```

### 4. Form Analysis

```python
# Simulate with different form levels
from src.simulator import PREMIER_LEAGUE_TEAMS

for form in [5.0, 6.5, 8.0, 9.5]:
    PREMIER_LEAGUE_TEAMS["Arsenal"].form = form
    
    matches = FootballMatchSimulator(seed=42).simulate_season(
        team="Arsenal",
        num_matches=20
    )
    
    goals = sum(m.home_score if m.home_team == "Arsenal" 
                else m.away_score for m in matches)
    
    print(f"Form {form}: {goals} goals in 20 matches")
```

## Limitations

### What the Simulator Does Well
- Realistic score distributions
- Appropriate statistical correlations
- Balanced home/away performance
- Sensible possession and shot numbers

### What the Simulator Doesn't Capture
- Player injuries and suspensions
- Tactical variations
- Weather conditions (beyond basic metadata)
- Referee bias
- Specific player performances
- Exact event sequences (except as aggregated stats)

### Calibration

The model is calibrated to:
- Premier League 2023-24 season
- Average ~2.8 goals per match
- Home win ~46%, Draw ~26%, Away win ~28%
- Typical possession ranges
- Realistic shot and xG distributions

## Tips and Best Practices

### 1. Use Seeds for Reproducibility

Always use seeds when:
- Creating training datasets
- Running experiments
- Comparing scenarios

```python
# Good
dataset = create_simulated_dataset(seed=42)

# Can't reproduce
dataset = create_simulated_dataset()
```

### 2. Generate Large Datasets

For ML training, use many matches:

```python
# Too small for robust training
dataset = create_simulated_dataset(num_matches=50)

# Better for training
dataset = create_simulated_dataset(num_matches=1000)

# Even better - multiple seasons/seeds
datasets = [create_simulated_dataset(num_matches=500, seed=s) 
            for s in range(10)]
```

### 3. Combine Real and Simulated Data

Use simulation to augment real data:

```python
# Load real data
real_dataset = load_real_data()

# Generate simulated data
sim_dataset = create_simulated_dataset(num_matches=500, seed=42)

# Combine for training
combined = real_dataset.matches + sim_dataset.matches
```

### 4. Validate Simulation Quality

Check that simulated data looks realistic:

```python
import pandas as pd

df = pd.read_csv("data/processed/Simulated_Arsenal_Matches_2023-24.csv")

# Check score distribution
print(df[['home_score', 'away_score']].describe())

# Check possession range
print(f"Possession range: {df['home_possession'].min()}-{df['home_possession'].max()}%")

# Check xG correlation with goals
correlation = df[['home_xg', 'home_score']].corr()
print(f"xG-Goals correlation: {correlation.iloc[0, 1]:.3f}")
```

## Troubleshooting

### Issue: Results seem unrealistic

**Solution**: Check team profiles and adjust if needed

```python
from src.simulator import PREMIER_LEAGUE_TEAMS

# View current profile
arsenal = PREMIER_LEAGUE_TEAMS["Arsenal"]
print(f"Arsenal - Attack: {arsenal.attack_strength}, Defense: {arsenal.defense_strength}")

# Adjust if needed
arsenal.attack_strength = 90  # Increase attack
```

### Issue: Too many high-scoring games

**Solution**: Reduce attack strengths or increase defense

### Issue: Not enough variation

**Solution**: Don't use seeds, or use different seeds

```python
# More variation
datasets = [create_simulated_dataset(seed=s) for s in range(10)]
```

## Future Enhancements

Potential improvements to the simulator:

- [ ] Player-level simulation
- [ ] Tactical system modeling
- [ ] Injury impact
- [ ] Weather effects
- [ ] Historical form tracking
- [ ] Derby/rivalry intensity
- [ ] Manager effects

## Support

For issues with the simulator:
1. Check team profiles in `src/simulator.py`
2. Verify your Python environment has numpy
3. Try different random seeds
4. Open an issue on GitHub with your simulation parameters

## References

- [Expected Goals (xG) Explained](https://fbref.com/en/expected-goals-model-explained/)
- [Poisson Distribution in Football](https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/)
- [Football Statistics Analysis](https://statsbomb.com/)
