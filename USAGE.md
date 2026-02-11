# Usage Guide

This guide provides detailed instructions for using the Arsenal FC match data collection tools.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jawaharganesh24189/Gunnersforeve.git
cd Gunnersforeve
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up API keys:
```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

## Collecting Data

### Using the Collection Script

The `fetch_data.py` script is the main tool for collecting match data.

#### Basic Usage (Mock Data)

Without an API key, the script will use mock data for demonstration:

```bash
python scripts/fetch_data.py --season 2023
```

#### With API Key

To fetch real data, provide an API key:

```bash
# Via command line
python scripts/fetch_data.py --api-key YOUR_KEY --season 2023

# Via environment variable
export FOOTBALL_DATA_API_KEY=your_key_here
python scripts/fetch_data.py --season 2023

# Via .env file
# Add FOOTBALL_DATA_API_KEY=your_key_here to .env
python scripts/fetch_data.py --season 2023
```

#### Output Formats

```bash
# JSON only
python scripts/fetch_data.py --season 2023 --format json

# CSV only
python scripts/fetch_data.py --season 2023 --format csv

# Both formats (default)
python scripts/fetch_data.py --season 2023 --format both
```

#### Custom Output Directory

```bash
python scripts/fetch_data.py --season 2023 --output-dir my_data/
```

### Programmatic Usage

You can also use the data collection functions in your own Python scripts:

```python
from src.data_collector import create_dataset_from_api, save_dataset

# Create a dataset
dataset = create_dataset_from_api(api_key="your_key", season="2023")

# Access the data
print(f"Collected {len(dataset.matches)} matches")
for match in dataset.matches:
    print(f"{match.date}: {match.home_team} vs {match.away_team}")

# Save the dataset
save_dataset(dataset, format="json", output_dir="data/processed")
save_dataset(dataset, format="csv", output_dir="data/processed")
```

## Working with Datasets

### Loading JSON Data

```python
import json
from src.data_schema import Dataset

# Load dataset
with open('data/examples/arsenal_matches_sample.json', 'r') as f:
    data = json.load(f)
    dataset = Dataset(**data)

# Access match information
for match in dataset.matches:
    arsenal_score = match.home_score if match.is_arsenal_home else match.away_score
    opponent_score = match.away_score if match.is_arsenal_home else match.home_score
    result = "Win" if arsenal_score > opponent_score else ("Draw" if arsenal_score == opponent_score else "Loss")
    
    print(f"{match.date} - Arsenal {arsenal_score}-{opponent_score} {match.away_team if match.is_arsenal_home else match.home_team} ({result})")
```

### Loading CSV Data with Pandas

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Calculate Arsenal's results
def get_result(row):
    if row['is_arsenal_home']:
        arsenal_score = row['home_score']
        opponent_score = row['away_score']
    else:
        arsenal_score = row['away_score']
        opponent_score = row['home_score']
    
    if arsenal_score > opponent_score:
        return 'Win'
    elif arsenal_score == opponent_score:
        return 'Draw'
    else:
        return 'Loss'

df['result'] = df.apply(get_result, axis=1)

# Statistics
print(f"Matches played: {len(df)}")
print(f"Wins: {(df['result'] == 'Win').sum()}")
print(f"Draws: {(df['result'] == 'Draw').sum()}")
print(f"Losses: {(df['result'] == 'Loss').sum()}")

# Goals scored and conceded
arsenal_goals = df.apply(
    lambda row: row['home_score'] if row['is_arsenal_home'] else row['away_score'], 
    axis=1
).sum()
opponent_goals = df.apply(
    lambda row: row['away_score'] if row['is_arsenal_home'] else row['home_score'], 
    axis=1
).sum()

print(f"Goals scored: {arsenal_goals}")
print(f"Goals conceded: {opponent_goals}")
print(f"Goal difference: {arsenal_goals - opponent_goals}")
```

## Data Analysis Examples

### Basic Statistics

```python
import pandas as pd

df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Home vs Away performance
home_matches = df[df['is_arsenal_home']]
away_matches = df[~df['is_arsenal_home']]

print("Home Stats:")
print(f"  Matches: {len(home_matches)}")
print(f"  Goals/match: {home_matches['home_score'].mean():.2f}")
print(f"  Possession: {home_matches['home_possession'].mean():.1f}%")

print("\nAway Stats:")
print(f"  Matches: {len(away_matches)}")
print(f"  Goals/match: {away_matches['away_score'].mean():.2f}")
print(f"  Possession: {away_matches['away_possession'].mean():.1f}%")
```

### Advanced Metrics

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Calculate xG (Expected Goals) for Arsenal
arsenal_xg = df.apply(
    lambda row: row['home_xg'] if row['is_arsenal_home'] else row['away_xg'],
    axis=1
)

arsenal_goals = df.apply(
    lambda row: row['home_score'] if row['is_arsenal_home'] else row['away_score'],
    axis=1
)

# Performance vs expectation
xg_diff = arsenal_goals - arsenal_xg
print(f"Total xG: {arsenal_xg.sum():.2f}")
print(f"Total goals: {arsenal_goals.sum()}")
print(f"Overperformance: {xg_diff.sum():.2f} goals")
```

## Data Validation

All data follows a strict schema defined in `src/data_schema.py`. You can validate data using Pydantic:

```python
from src.data_schema import MatchData, TeamStats

# Validate match data
match = MatchData(
    date="2023-08-12",
    competition="Premier League",
    season="2023-24",
    home_team="Arsenal",
    away_team="Chelsea",
    is_arsenal_home=True,
    home_score=2,
    away_score=1,
    home_stats=TeamStats(
        team_name="Arsenal",
        goals=2,
        shots=15,
        possession=60.0
    )
)

print(match.model_dump_json(indent=2))
```

## Machine Learning Applications

The standardized datasets are ready for ML applications:

### Feature Engineering

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Create features
features = df[[
    'home_possession', 'home_shots', 'home_shots_on_target',
    'home_corners', 'home_xg'
]].copy()

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Prediction Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/examples/arsenal_matches_sample.csv')

# Create target (Win/Draw/Loss)
def get_result(row):
    if row['is_arsenal_home']:
        diff = row['home_score'] - row['away_score']
    else:
        diff = row['away_score'] - row['home_score']
    
    if diff > 0:
        return 2  # Win
    elif diff == 0:
        return 1  # Draw
    else:
        return 0  # Loss

df['result'] = df.apply(get_result, axis=1)

# Select features
feature_cols = ['home_possession', 'home_shots', 'home_corners']
X = df[feature_cols]
y = df['result']

# Note: This is a minimal example. Real ML applications would need:
# - More data
# - Feature engineering
# - Proper train/test split
# - Model validation
```

## API Reference

### FootballDataCollector

```python
from src.data_collector import FootballDataCollector

collector = FootballDataCollector(api_key="your_key")
matches = collector.get_matches(season="2023", limit=100)
```

### Dataset Operations

```python
from src.data_schema import Dataset

# Export to CSV
dataset.to_csv('output.csv')

# Access metadata
print(dataset.dataset_name)
print(dataset.source)
print(f"Last updated: {dataset.last_updated}")
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running scripts from the repository root:

```bash
cd /path/to/Gunnersforeve
python scripts/fetch_data.py
```

### API Rate Limits

If you hit API rate limits:
- Use cached data from previous runs
- Reduce the frequency of requests
- Consider upgrading to a paid API tier

### Data Quality Issues

To check data quality:
```python
import pandas as pd

df = pd.read_csv('your_data.csv')

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Validate scores are non-negative
assert (df['home_score'] >= 0).all()
assert (df['away_score'] >= 0).all()
```

## Additional Resources

- [DATA_SOURCES.md](DATA_SOURCES.md) - Complete list of data sources
- [football-data.org Documentation](https://www.football-data.org/documentation/quickstart)
- [FBref Glossary](https://fbref.com/en/expected-goals-model-explained/)

## Support

For issues or questions:
1. Check this guide and the README
2. Review example datasets in `data/examples/`
3. Open an issue on GitHub
