# Tactical Match Simulation Guide

## Overview

The Advanced Tactical Simulator (`tactical_simulator.py`) provides sophisticated, event-level match simulation with realistic tactical dynamics. Unlike the basic simulator which generates final statistics, the tactical simulator simulates matches **minute-by-minute** with formations, playing styles, and dynamic match events.

## Key Features

### 1. Event-Level Simulation
- Minute-by-minute match progression
- Detailed event log with timestamps
- Track every shot, goal, foul, card, corner, save
- Watch momentum shifts in real-time

### 2. Tactical Depth
- **Formations**: 4-3-3, 4-4-2, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2
- **Playing Styles**: Possession, Counter-attack, High Press, Defensive, Balanced, Direct
- **Tactical Setup**: Line height, pressing intensity, width, tempo
- **Half-time Adjustments**: Teams adapt tactics based on score

### 3. Match Dynamics
- **Momentum**: Shifts based on goals and play
- **Energy**: Decreases over match, affects performance
- **Morale**: Changes with goals scored/conceded
- **Possession**: Calculated per minute based on tactics

### 4. Advanced Team Profiles

Each team has 10+ attributes:
- **Core**: Attack, Defense, Midfield strength
- **Tactical**: Pressing ability, passing quality, pace, physicality, creativity, discipline
- **Situational**: Form, home advantage
- **Preferences**: Preferred formation and playing style

## Usage

### Command Line

#### Basic Tactical Match

```bash
# Simulate Arsenal vs Manchester City with detailed events
python scripts/simulate_tactical.py Arsenal "Manchester City" --detailed

# Show minute-by-minute event log
python scripts/simulate_tactical.py Arsenal Liverpool --detailed --show-events

# Quick simulation (no events, faster)
python scripts/simulate_tactical.py Arsenal Chelsea --quick
```

#### Multiple Matches

```bash
# Simulate 10 matches quickly
python scripts/simulate_tactical.py Arsenal Tottenham --quick --matches 10

# Simulate series with detailed events (slower)
python scripts/simulate_tactical.py Arsenal Liverpool --detailed --matches 5
```

#### Save Results

```bash
# Simulate and save to file
python scripts/simulate_tactical.py Arsenal "Manchester City" --save

# Multiple matches with save
python scripts/simulate_tactical.py Arsenal Chelsea --matches 10 --save
```

### Python API

#### Simple Match Simulation

```python
from src.tactical_simulator import TacticalMatchSimulator

# Create simulator
simulator = TacticalMatchSimulator(seed=42)

# Simulate match with detailed events
match_data, match_state = simulator.simulate_tactical_match(
    home_team="Arsenal",
    away_team="Manchester City",
    date="2024-03-31",
    detailed_events=True
)

# Print summary
print(simulator.get_match_summary())

# Access event log
for event in match_state.events:
    print(f"{event.minute}' - {event.description}")
```

#### Access Match State

```python
# Get match state attributes
print(f"Final Score: {match_state.home_score}-{match_state.away_score}")
print(f"Possession: {match_state.home_possession:.1f}% - {match_state.away_possession:.1f}%")
print(f"Momentum: {match_state.momentum}")  # -100 to 100
print(f"Shots: {match_state.home_shots} - {match_state.away_shots}")
print(f"Yellow cards: {match_state.home_yellow_cards} - {match_state.away_yellow_cards}")

# Get specific events
goals = [e for e in match_state.events if e.event_type.value == "goal"]
for goal in goals:
    print(f"{goal.minute}' - GOAL by {goal.team}")
```

#### Quick Simulation for Datasets

```python
# For large datasets, use quick mode (no event tracking)
simulator = TacticalMatchSimulator(seed=42)

matches = []
for i in range(100):
    match_data, match_state = simulator.simulate_tactical_match(
        home_team="Arsenal",
        away_team="Liverpool",
        date=f"2024-01-{i+1:02d}",
        detailed_events=False  # Much faster
    )
    matches.append(match_data)
```

## Team Profiles

### Current Premier League Teams

The simulator includes detailed profiles for:

**Top Teams:**
- **Arsenal**: 4-3-3, Possession style (Attack: 88, Defense: 82, Midfield: 86)
- **Manchester City**: 4-3-3, Possession style (Attack: 92, Defense: 85, Midfield: 90)
- **Liverpool**: 4-3-3, High Press style (Attack: 90, Defense: 80, Midfield: 87)

**Mid-Table:**
- **Manchester United**: 4-2-3-1, Counter-attack (Attack: 78, Defense: 72, Midfield: 75)
- **Chelsea**: 4-2-3-1, Balanced (Attack: 80, Defense: 75, Midfield: 78)
- **Tottenham**: 4-2-3-1, Counter-attack (Attack: 82, Defense: 70, Midfield: 76)
- **Newcastle**: 4-3-3, High Press (Attack: 77, Defense: 80, Midfield: 78)

### View Team Profile

```python
from src.tactical_simulator import ADVANCED_PL_TEAMS

# Get Arsenal profile
arsenal = ADVANCED_PL_TEAMS["Arsenal"]

print(f"Team: {arsenal.name}")
print(f"Overall Strength: {arsenal.overall_strength:.1f}")
print(f"Formation: {arsenal.preferred_formation.value}")
print(f"Style: {arsenal.preferred_style.value}")
print(f"\nCore Attributes:")
print(f"  Attack: {arsenal.attack_strength}")
print(f"  Defense: {arsenal.defense_strength}")
print(f"  Midfield: {arsenal.midfield_strength}")
print(f"\nTactical Attributes:")
print(f"  Pressing: {arsenal.pressing_ability}")
print(f"  Passing: {arsenal.passing_quality}")
print(f"  Pace: {arsenal.pace}")
print(f"  Creativity: {arsenal.creativity}")
print(f"  Discipline: {arsenal.discipline}")
```

## Playing Styles

### Possession
- High passing quality teams (Manchester City, Arsenal)
- Control tempo, dominate possession
- Patient build-up play
- Higher defensive line

### High Press
- High energy, aggressive pressing (Liverpool, Newcastle)
- Win ball high up the pitch
- Quick transitions
- Very high defensive line

### Counter-Attack
- Fast, direct play (Tottenham, Manchester United)
- Absorb pressure, hit on break
- Pace is crucial
- Moderate defensive line

### Defensive
- Compact, organized defending
- Low defensive line
- Limited possession
- Rely on set pieces

### Balanced
- Adaptable approach (Chelsea)
- Mix of possession and directness
- Medium defensive line

## Match Dynamics

### Momentum

Momentum ranges from -100 (away dominance) to +100 (home dominance):

```python
if match_state.momentum > 30:
    print("Home team dominating!")
elif match_state.momentum < -30:
    print("Away team on top!")
else:
    print("Evenly matched")
```

### Energy

Energy starts at 100% and decreases throughout the match:
- Affects attacking and defensive quality
- Teams with higher physicality maintain energy better
- Drops faster with high pressing intensity

### Morale

Morale changes with goals:
- Scoring increases morale (+15)
- Conceding decreases morale (-10)
- Affects team confidence and performance

## Event Types

The simulator tracks these event types:

| Event Type | Description |
|------------|-------------|
| `kick_off` | Match/half starts |
| `pass` | Successful/unsuccessful pass |
| `shot` | Shot attempt |
| `goal` | Goal scored ⚽ |
| `save` | Goalkeeper save |
| `corner` | Corner kick awarded |
| `free_kick` | Free kick awarded |
| `tackle` | Tackle made |
| `foul` | Foul committed |
| `yellow_card` | Yellow card shown 🟨 |
| `red_card` | Red card shown 🟥 |
| `substitution` | Player substitution |
| `half_time` | Half-time break |
| `full_time` | Match ends |

## Advanced Usage

### Filter Events

```python
# Get only goals
goals = [e for e in match_state.events 
         if e.event_type.value == "goal"]

# Get all cards
cards = [e for e in match_state.events 
         if e.event_type.value in ["yellow_card", "red_card"]]

# Get first half events
first_half = [e for e in match_state.events 
              if e.minute <= 45]
```

### Statistical Analysis

```python
# Analyze goal timing
goal_minutes = [e.minute for e in match_state.events 
                if e.event_type.value == "goal"]

import matplotlib.pyplot as plt
plt.hist(goal_minutes, bins=9, range=(0, 90))
plt.xlabel("Minute")
plt.ylabel("Goals")
plt.title("Goal Distribution by Minute")
plt.show()
```

### Compare Styles

```python
from src.tactical_simulator import TacticalMatchSimulator, ADVANCED_PL_TEAMS

simulator = TacticalMatchSimulator(seed=42)

# Compare possession vs counter-attack
possession_team = "Arsenal"  # Possession style
counter_team = "Tottenham"   # Counter-attack style

results = {"possession_wins": 0, "counter_wins": 0, "draws": 0}

for i in range(100):
    match, state = simulator.simulate_tactical_match(
        possession_team, counter_team, 
        f"2024-01-{i+1:02d}", 
        detailed_events=False
    )
    
    if match.home_score > match.away_score:
        results["possession_wins"] += 1
    elif match.home_score < match.away_score:
        results["counter_wins"] += 1
    else:
        results["draws"] += 1

print(results)
```

## Performance

### Detailed vs Quick Mode

**Detailed Mode** (minute-by-minute):
- Full event log
- ~0.1-0.2 seconds per match
- Use for: Single matches, analysis, visualization

**Quick Mode** (statistical):
- No event log
- ~0.01-0.02 seconds per match
- Use for: Large datasets, batch generation, training data

### Optimization Tips

```python
# Generate 1000 matches quickly
simulator = TacticalMatchSimulator(seed=42)

matches = []
for i in range(1000):
    match_data, _ = simulator.simulate_tactical_match(
        "Arsenal", "Liverpool",
        f"2024-{i//30+1:02d}-{i%30+1:02d}",
        detailed_events=False  # 10x faster
    )
    matches.append(match_data)
```

## Comparison: Basic vs Tactical Simulator

| Feature | Basic Simulator | Tactical Simulator |
|---------|----------------|-------------------|
| Speed | Fast (~0.01s/match) | Slower detailed (~0.15s/match), Fast quick (~0.02s/match) |
| Event Log | No | Yes (detailed mode) |
| Formations | No | Yes (6 formations) |
| Playing Styles | No | Yes (6 styles) |
| Match Dynamics | No | Yes (momentum, energy, morale) |
| Tactical Adjustments | No | Yes (half-time changes) |
| Team Attributes | 3 (ATT, DEF, MID) | 10+ (tactical depth) |
| Best For | Large datasets | Realistic simulations, analysis |

## Examples

### Simulate a Derby

```python
# North London Derby with full details
simulator = TacticalMatchSimulator(seed=42)

match_data, match_state = simulator.simulate_tactical_match(
    home_team="Arsenal",
    away_team="Tottenham",
    date="2024-09-15",
    detailed_events=True
)

print(simulator.get_match_summary())

# Print all goals with minute
goals = [e for e in match_state.events if e.event_type.value == "goal"]
print(f"\n{len(goals)} goals scored:")
for goal in goals:
    print(f"  {goal.minute}' - {goal.team}")
```

### Analyze Possession Dominance

```python
# Does possession lead to goals?
import pandas as pd

simulator = TacticalMatchSimulator(seed=42)

data = []
for i in range(200):
    match_data, match_state = simulator.simulate_tactical_match(
        "Arsenal", "Chelsea",
        f"2024-{i//30+1:02d}-{i%30+1:02d}",
        detailed_events=False
    )
    
    data.append({
        'possession': match_state.home_possession,
        'goals': match_state.home_score,
        'result': 'W' if match_state.home_score > match_state.away_score else 'L'
    })

df = pd.DataFrame(data)
print(f"Possession vs Goals correlation: {df['possession'].corr(df['goals']):.3f}")
```

## Tips

1. **Use `seed` for reproducibility** in research and testing
2. **Use detailed mode** for single match analysis
3. **Use quick mode** for generating large training datasets
4. **Adjust team profiles** for "what-if" scenarios
5. **Filter events** to focus on specific moments
6. **Analyze patterns** across multiple simulations

## Future Enhancements

Potential additions:
- Individual player simulation
- Injury modeling
- Weather effects
- Substitution AI
- Set piece specialists
- Manager influence
- Cup competition modeling

## Support

For questions or issues:
1. Check this guide
2. See `tactical_simulator.py` source code
3. Run examples in the repository
4. Open an issue on GitHub
