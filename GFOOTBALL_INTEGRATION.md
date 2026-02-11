# Google Research Football Integration Example

This file demonstrates how to integrate Google Research Football with the tactical AI model from the main notebook.

## Installation

```bash
pip install gfootball
```

## Basic Usage

```python
import gfootball.env as football_env
import numpy as np

# Create environment with different scenarios
env = football_env.create_environment(
    env_name='academy_empty_goal_close',  # or '11_vs_11_stochastic', 'academy_pass_and_shoot_with_keeper'
    representation='raw',  # Get full observation data
    render=True,  # Set to False for faster training
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    logdir='/tmp/football'
)

# Run a simple episode
obs = env.reset()
done = False
total_reward = 0

while not done:
    # Random action (replace with your AI model)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Episode finished with reward: {total_reward}")
```

## Extracting Tracking Data from GFootball

```python
def extract_tracking_data_from_gfootball(env, num_episodes=10):
    """
    Extract tracking data from Google Research Football environment.
    Returns data in format compatible with the BiLSTM + Attention model.
    """
    all_tracking_data = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_tracking = []
        done = False
        
        while not done:
            # Get current observation (raw format includes all player positions)
            if isinstance(obs, dict):
                # Extract player positions
                left_team = obs.get('left_team', np.zeros((11, 2)))
                right_team = obs.get('right_team', np.zeros((11, 2)))
                ball_pos = obs.get('ball', np.zeros(3))[:2]  # x, y only
                
                # Calculate distances to goal
                goal_pos = np.array([1.0, 0.0])  # Normalized goal position
                distances = np.linalg.norm(left_team - goal_pos, axis=1)
                
                # Store tracking info
                sequence_data = {
                    'positions': left_team.copy(),  # Shape: (11, 2)
                    'ball_position': ball_pos,
                    'opponent_positions': right_team.copy(),
                    'distances_to_goal': distances,
                    'is_goal': 0  # Will be updated if goal scored
                }
                episode_tracking.append(sequence_data)
            
            # Take action (replace with your model)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Mark goal sequences
            if reward > 0 and len(episode_tracking) > 0:
                # Mark last N frames as goal sequences
                for i in range(max(0, len(episode_tracking) - 10), len(episode_tracking)):
                    episode_tracking[i]['is_goal'] = 1
        
        all_tracking_data.extend(episode_tracking)
    
    return all_tracking_data

# Collect data
tracking_data = extract_tracking_data_from_gfootball(env, num_episodes=50)
print(f"Collected {len(tracking_data)} tracking sequences")
```

## Training Your Model on GFootball Data

```python
def prepare_gfootball_data_for_model(tracking_data, sequence_length=20):
    """
    Convert GFootball tracking data to format for BiLSTM + Attention model.
    """
    X_positions = []
    X_features = []
    y_labels = []
    
    # Group into sequences
    for i in range(0, len(tracking_data) - sequence_length):
        sequence = tracking_data[i:i + sequence_length]
        
        # Extract positions for each timestep
        positions = np.array([s['positions'].flatten() for s in sequence])  # (seq_len, 22)
        
        # Extract additional features
        features = np.array([
            [
                s['ball_position'][0],
                s['ball_position'][1],
                s['distances_to_goal'].mean(),
                s['distances_to_goal'].min()
            ]
            for s in sequence
        ])  # (seq_len, 4)
        
        # Label is whether this sequence leads to a goal
        label = sequence[-1]['is_goal']
        
        X_positions.append(positions)
        X_features.append(features)
        y_labels.append(label)
    
    return np.array(X_positions), np.array(X_features), np.array(y_labels)

# Prepare data
X_pos, X_feat, y = prepare_gfootball_data_for_model(tracking_data)

print(f"Training data shapes:")
print(f"  X_positions: {X_pos.shape}")
print(f"  X_features: {X_feat.shape}")
print(f"  y: {y.shape}")

# Now you can train your BiLSTM + Attention model with this data
# model.fit([X_pos, X_feat], y, epochs=20, validation_split=0.2)
```

## Available Scenarios

Google Research Football provides many pre-built scenarios:

### Academy Scenarios (Training/Drills)
- `academy_empty_goal_close` - Score in empty goal from close range
- `academy_empty_goal` - Score in empty goal from various positions
- `academy_run_to_score` - Run with ball and score
- `academy_run_to_score_with_keeper` - Score against goalkeeper
- `academy_pass_and_shoot_with_keeper` - Pass and shoot drill
- `academy_3_vs_1_with_keeper` - 3v1 attacking scenario
- `academy_corner` - Corner kick scenario
- `academy_counterattack_easy` - Counter-attack situation

### Full Game Scenarios
- `11_vs_11_kaggle` - Full 11v11 match (Kaggle competition version)
- `11_vs_11_easy_stochastic` - Full match with some randomness
- `11_vs_11_hard_stochastic` - Challenging full match
- `11_vs_11_competition` - Competition-grade match

## Action Space

GFootball provides different action sets:

- `default` (19 actions): All basic actions
- `full` (19 actions): Same as default
- `simple` (21 actions): Simplified control

Actions include: idle, move (8 directions), sprint, dribble, shot, pass (3 types), tackle, slide, keeper actions.

## Observation Representations

- `simple115` - Simple feature vector (115 dimensions)
- `simple115v2` - Enhanced simple features
- `extracted` - Spatial representation for CNNs
- `pixels` - Raw pixel screen (for vision-based learning)
- `raw` - Full game state (all positions, stats) - Best for your use case

## Example: Complete Training Pipeline

```python
import gfootball.env as football_env
from your_notebook import build_tactical_ai_model  # Import from main notebook

# 1. Create environment
env = football_env.create_environment(
    env_name='academy_pass_and_shoot_with_keeper',
    representation='raw',
    render=False
)

# 2. Collect training data
print("Collecting data from GFootball...")
tracking_data = extract_tracking_data_from_gfootball(env, num_episodes=100)

# 3. Prepare for model
X_pos, X_feat, y = prepare_gfootball_data_for_model(tracking_data)

# 4. Build and train model
model = build_tactical_ai_model(
    sequence_length=X_pos.shape[1],
    pos_features=X_pos.shape[2],
    add_features=X_feat.shape[2]
)

# 5. Train
history = model.fit(
    [X_pos, X_feat],
    y,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# 6. Evaluate in environment
obs = env.reset()
done = False
while not done:
    # Use your model to predict best action
    # (You'll need to convert obs to model input format)
    action = your_action_selection_logic(model, obs)
    obs, reward, done, info = env.step(action)
```

## Benefits of GFootball Integration

1. **Realistic Physics**: Ball dynamics, player collisions, stamina
2. **Rich Observations**: Complete game state with all player positions
3. **Diverse Scenarios**: From simple drills to full 11v11 matches
4. **Active Development**: Regular updates and community support
5. **Research Grade**: Used in academic papers and competitions
6. **Pre-trained Baselines**: Compare your model against existing agents

## Performance Tips

1. Use `render=False` for faster training
2. Start with academy scenarios before full matches
3. Use `stacked=True` for temporal information
4. Consider `representation='extracted'` for CNN-based models
5. Save checkpoints regularly
6. Use multiple parallel environments for faster data collection

## Next Steps

1. Install GFootball: `pip install gfootball`
2. Run this example code in a new notebook cell
3. Collect data from various scenarios
4. Train your BiLSTM + Attention model
5. Compare simple simulation vs GFootball performance
6. Fine-tune your model on specific scenarios
