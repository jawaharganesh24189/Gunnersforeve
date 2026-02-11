"""
Visualization utilities for football tactics transformer.

This module provides functions to visualize:
- Training metrics (loss and accuracy curves)
- Tactical formations on a football pitch
- Passing sequences
- Model predictions vs actual data
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional


def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history_path: Path to training history JSON file
        save_path: Optional path to save the figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['masked_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_masked_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def draw_football_pitch(ax, pitch_color='#195905', line_color='white'):
    """
    Draw a football pitch on the given axes.
    
    Args:
        ax: Matplotlib axes object
        pitch_color: Color of the pitch
        line_color: Color of the lines
    """
    # Set pitch background
    ax.set_facecolor(pitch_color)
    
    # Pitch outline
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=line_color, linewidth=2)
    
    # Halfway line
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=2)
    
    # Center circle
    circle = plt.Circle((50, 50), 9.15, fill=False, color=line_color, linewidth=2)
    ax.add_patch(circle)
    
    # Center spot
    ax.plot(50, 50, 'o', color=line_color, markersize=3)
    
    # Left penalty area
    ax.plot([0, 16.5, 16.5, 0], [21.1, 21.1, 78.9, 78.9], color=line_color, linewidth=2)
    
    # Right penalty area
    ax.plot([100, 83.5, 83.5, 100], [21.1, 21.1, 78.9, 78.9], color=line_color, linewidth=2)
    
    # Left goal area
    ax.plot([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2], color=line_color, linewidth=2)
    
    # Right goal area
    ax.plot([100, 94.5, 94.5, 100], [36.8, 36.8, 63.2, 63.2], color=line_color, linewidth=2)
    
    # Left penalty arc
    arc_left = patches.Arc((11, 50), 18.3, 18.3, angle=0, theta1=310, theta2=50, 
                           color=line_color, linewidth=2)
    ax.add_patch(arc_left)
    
    # Right penalty arc
    arc_right = patches.Arc((89, 50), 18.3, 18.3, angle=0, theta1=130, theta2=230, 
                            color=line_color, linewidth=2)
    ax.add_patch(arc_right)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set equal aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)


def plot_formation(
    formation: str,
    team_name: str = "Team",
    save_path: Optional[str] = None
):
    """
    Visualize a team formation on a football pitch.
    
    Args:
        formation: Formation string (e.g., '4-3-3')
        team_name: Name of the team
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 15))
    draw_football_pitch(ax)
    
    # Formation positions (x, y) - simplified layout
    formation_positions = {
        '4-3-3': [
            (5, 50),    # GK
            (20, 20), (20, 40), (20, 60), (20, 80),  # Defense
            (45, 25), (45, 50), (45, 75),  # Midfield
            (75, 25), (75, 50), (75, 75),  # Attack
        ],
        '4-4-2': [
            (5, 50),    # GK
            (20, 20), (20, 40), (20, 60), (20, 80),  # Defense
            (45, 20), (45, 40), (45, 60), (45, 80),  # Midfield
            (75, 35), (75, 65),  # Attack
        ],
        '3-5-2': [
            (5, 50),    # GK
            (20, 25), (20, 50), (20, 75),  # Defense
            (40, 15), (40, 35), (40, 50), (40, 65), (40, 85),  # Midfield
            (75, 35), (75, 65),  # Attack
        ],
        '4-2-3-1': [
            (5, 50),    # GK
            (20, 20), (20, 40), (20, 60), (20, 80),  # Defense
            (40, 35), (40, 65),  # Defensive Midfield
            (55, 25), (55, 50), (55, 75),  # Attacking Midfield
            (80, 50),  # Striker
        ],
        '3-4-3': [
            (5, 50),    # GK
            (20, 25), (20, 50), (20, 75),  # Defense
            (45, 20), (45, 40), (45, 60), (45, 80),  # Midfield
            (75, 25), (75, 50), (75, 75),  # Attack
        ],
    }
    
    positions = formation_positions.get(formation, formation_positions['4-3-3'])
    
    # Plot players
    for x, y in positions:
        circle = plt.Circle((x, y), 3, color='red', zorder=10)
        ax.add_patch(circle)
        ax.plot(x, y, 'o', color='white', markersize=5, zorder=11)
    
    ax.set_title(f'{team_name} Formation: {formation}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Formation plot saved to {save_path}")
    
    return fig


def plot_passing_sequence(
    sequence: List[Tuple[str, str]],
    title: str = "Passing Sequence",
    save_path: Optional[str] = None
):
    """
    Visualize a passing sequence on a football pitch.
    
    Args:
        sequence: List of (position, action) tuples
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 15))
    draw_football_pitch(ax)
    
    # Position coordinates mapping (simplified)
    position_coords = {
        'GK': (5, 50),
        'CB': (20, 50), 'LB': (20, 20), 'RB': (20, 80),
        'LWB': (25, 15), 'RWB': (25, 85),
        'CDM': (35, 50), 'CM': (45, 50),
        'LM': (45, 25), 'RM': (45, 75),
        'CAM': (60, 50),
        'LW': (70, 25), 'RW': (70, 75),
        'ST': (85, 50), 'CF': (85, 50),
    }
    
    # Plot sequence
    coords = []
    for i, (position, action) in enumerate(sequence):
        if position in position_coords:
            x, y = position_coords[position]
            # Add some randomness to avoid overlap
            x += np.random.randint(-3, 4)
            y += np.random.randint(-3, 4)
            coords.append((x, y))
            
            # Plot player
            circle = plt.Circle((x, y), 2.5, color='blue', zorder=10, alpha=0.7)
            ax.add_patch(circle)
            
            # Add label
            ax.text(x, y - 6, f"{i+1}. {position}", 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Draw arrows between positions
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='yellow', alpha=0.8))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Passing sequence plot saved to {save_path}")
    
    return fig


def plot_model_summary(
    config_path: str,
    history_path: str,
    save_dir: str
):
    """
    Create a comprehensive visualization of model training and architecture.
    
    Args:
        config_path: Path to model configuration JSON
        history_path: Path to training history JSON
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['masked_accuracy'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['val_masked_accuracy'], 'r-', label='Validation', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Configuration
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    config_text = f"""
    MODEL CONFIGURATION
    {'=' * 60}
    
    Architecture Parameters:
    • Number of Layers: {config['num_layers']}
    • Model Dimension (d_model): {config['d_model']}
    • Number of Attention Heads: {config['num_heads']}
    • Feed-Forward Dimension: {config['dff']}
    • Dropout Rate: {config['dropout_rate']}
    
    Vocabulary Sizes:
    • Input Vocabulary: {config['input_vocab_size']}
    • Target Vocabulary: {config['target_vocab_size']}
    • Max Position Encoding: {config['max_position_encoding']}
    
    Training Results:
    • Final Training Loss: {history['loss'][-1]:.4f}
    • Final Validation Loss: {history['val_loss'][-1]:.4f}
    • Final Training Accuracy: {history['masked_accuracy'][-1]:.4f}
    • Final Validation Accuracy: {history['val_masked_accuracy'][-1]:.4f}
    • Total Epochs: {len(history['loss'])}
    """
    
    ax3.text(0.1, 0.5, config_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Learning Curves - Last 20 epochs
    ax4 = fig.add_subplot(gs[2, 0])
    last_n = min(20, len(history['loss']))
    last_epochs = range(len(epochs) - last_n + 1, len(epochs) + 1)
    ax4.plot(last_epochs, history['loss'][-last_n:], 'b-', label='Training', linewidth=2, marker='o')
    ax4.plot(last_epochs, history['val_loss'][-last_n:], 'r-', label='Validation', linewidth=2, marker='s')
    ax4.set_title(f'Loss - Last {last_n} Epochs', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy - Last 20 epochs
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(last_epochs, history['masked_accuracy'][-last_n:], 'b-', label='Training', linewidth=2, marker='o')
    ax5.plot(last_epochs, history['val_masked_accuracy'][-last_n:], 'r-', label='Validation', linewidth=2, marker='s')
    ax5.set_title(f'Accuracy - Last {last_n} Epochs', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Save figure
    summary_path = os.path.join(save_dir, 'model_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Model summary saved to {summary_path}")
    
    return fig


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Check if model files exist
    if os.path.exists('models/training_history.json'):
        print("Creating training history visualization...")
        plot_training_history(
            'models/training_history.json',
            'models/training_curves.png'
        )
    
    if os.path.exists('models/model_config.json') and os.path.exists('models/training_history.json'):
        print("Creating comprehensive model summary...")
        plot_model_summary(
            'models/model_config.json',
            'models/training_history.json',
            'models/visualizations'
        )
    
    # Example: Plot formations
    print("Creating formation visualizations...")
    os.makedirs('models/visualizations', exist_ok=True)
    
    formations = ['4-3-3', '4-4-2', '3-5-2', '4-2-3-1', '3-4-3']
    for formation in formations:
        plot_formation(
            formation,
            f"Team Formation",
            f'models/visualizations/formation_{formation.replace("-", "_")}.png'
        )
    
    # Example: Plot a passing sequence
    example_sequence = [
        ('CB', 'short_pass'),
        ('CDM', 'forward_pass'),
        ('CAM', 'through_ball'),
        ('ST', 'shot')
    ]
    plot_passing_sequence(
        example_sequence,
        "Example Passing Sequence: Build-up Play",
        'models/visualizations/passing_sequence_example.png'
    )
    
    print("\nAll visualizations created successfully!")
