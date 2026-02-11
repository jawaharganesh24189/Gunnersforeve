"""
Training script for the Tactics Transformer model using real match data.

This script trains the transformer model on actual match data from match_history.py,
integrating real formations, tactical contexts, and passing sequences.
"""

import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

from .transformer_model import create_tactics_transformer
from .data_preprocessing import TacticsEncoder
from .match_history import load_match_history, MatchData


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule for transformer training.
    Implements warmup and decay strategy.
    """
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        """Return configuration for serialization."""
        return {
            'd_model': float(self.d_model.numpy()),
            'warmup_steps': self.warmup_steps
        }


def masked_loss(real, pred):
    """
    Masked loss function that ignores padding tokens.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(real, pred):
    """
    Masked accuracy metric that ignores padding tokens.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    pred_ids = tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype)
    accuracies = tf.equal(real, pred_ids)
    
    mask = tf.cast(mask, dtype=tf.float32)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    
    return tf.reduce_sum(accuracies * mask) / tf.reduce_sum(mask)


def create_training_data_from_matches(
    matches: List[MatchData],
    encoder: TacticsEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data from real match data.
    
    Args:
        matches: List of MatchData objects
        encoder: TacticsEncoder instance
    
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    input_sequences = []
    target_sequences = []
    
    for match in matches:
        if match.passing_sequences is None:
            continue
        
        # Create input from match tactical situation
        for sequence in match.passing_sequences:
            # Encode tactical situation as input
            # Format: [formation, opp_formation, ball_x, ball_y, context, ...]
            encoded_input = []
            encoded_input.append(encoder.encode_formation(match.home_formation))
            encoded_input.append(encoder.encode_formation(match.away_formation))
            
            # Add ball position (start from defense)
            encoded_input.append(20)  # x position (defensive third)
            encoded_input.append(50)  # y position (center)
            
            # Add tactical context
            encoded_input.append(encoder.encode_tactical_context(match.tactical_context))
            
            # Add player positions from sequence
            for pos, action, success_rate in sequence[:5]:  # Take first 5 positions
                encoded_input.append(encoder.encode_position(pos))
                # Add dummy coordinates
                encoded_input.append(np.random.randint(0, 100))
                encoded_input.append(np.random.randint(0, 100))
            
            input_sequences.append(np.array(encoded_input, dtype=np.int32))
            
            # Encode passing sequence as target
            encoded_target = [encoder.actions['<START>']]
            for pos, action, success_rate in sequence:
                encoded_target.append(encoder.encode_position(pos))
                # Extract action name from tuple or use default
                action_name = action if isinstance(action, str) else 'short_pass'
                encoded_target.append(encoder.encode_action(action_name))
            encoded_target.append(encoder.actions['<END>'])
            
            target_sequences.append(np.array(encoded_target, dtype=np.int32))
    
    # Pad sequences to same length
    if len(input_sequences) == 0:
        raise ValueError("No training data could be created from matches")
    
    max_input_len = max(len(seq) for seq in input_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)
    
    padded_inputs = np.zeros((len(input_sequences), max_input_len), dtype=np.int32)
    padded_targets = np.zeros((len(target_sequences), max_target_len), dtype=np.int32)
    
    for i, (inp, tar) in enumerate(zip(input_sequences, target_sequences)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tar)] = tar
    
    return padded_inputs, padded_targets


def augment_match_data(
    matches: List[MatchData],
    encoder: TacticsEncoder,
    augmentation_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment match data by generating variations of tactical situations.
    
    Args:
        matches: List of MatchData objects
        encoder: TacticsEncoder instance
        augmentation_factor: Number of variations to generate per match
    
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    # First, get the base data
    base_inputs, base_targets = create_training_data_from_matches(matches, encoder)
    
    # Create augmented versions
    all_inputs = [base_inputs]
    all_targets = [base_targets]
    
    formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1', '3-4-3']
    contexts = ['counter_attack', 'possession', 'build_from_back', 'high_press']
    
    for _ in range(augmentation_factor - 1):
        augmented_inputs = []
        
        for base_input in base_inputs:
            # Copy and modify formation and context
            aug_input = base_input.copy()
            
            # Randomly change formations (first two elements)
            if np.random.random() > 0.5:
                aug_input[0] = encoder.encode_formation(np.random.choice(formations))
            if np.random.random() > 0.5:
                aug_input[1] = encoder.encode_formation(np.random.choice(formations))
            
            # Randomly change context
            if np.random.random() > 0.3:
                aug_input[4] = encoder.encode_tactical_context(np.random.choice(contexts))
            
            # Add small variations to positions
            for i in range(5, len(aug_input), 3):
                if i + 2 < len(aug_input):
                    # Add small noise to coordinates
                    aug_input[i + 1] = max(0, min(100, aug_input[i + 1] + np.random.randint(-10, 11)))
                    aug_input[i + 2] = max(0, min(100, aug_input[i + 2] + np.random.randint(-10, 11)))
            
            augmented_inputs.append(aug_input)
        
        all_inputs.append(np.array(augmented_inputs))
        all_targets.append(base_targets.copy())
    
    # Concatenate all augmented data
    final_inputs = np.vstack(all_inputs)
    final_targets = np.vstack(all_targets)
    
    return final_inputs, final_targets


def train_model_on_matches(
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512,
    dropout_rate=0.1,
    epochs=100,
    batch_size=16,
    save_dir='models',
    augmentation_factor=20
):
    """
    Train the tactics transformer model on real match data.
    
    Args:
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward network dimension
        dropout_rate: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save trained models
        augmentation_factor: Data augmentation multiplier
    
    Returns:
        Trained model and training history
    """
    print("Loading real match data...")
    loader = load_match_history()
    matches = loader.matches
    
    stats = loader.get_statistics()
    print(f"\nMatch Data Statistics:")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Average goals: {stats['avg_goals']:.2f}")
    print(f"  Average possession (home): {stats['avg_possession_home']:.1f}%")
    print(f"  Formations used: {', '.join(stats['formations'])}")
    
    # Initialize encoder
    encoder = TacticsEncoder()
    
    print("\nPreparing training data from real matches...")
    train_inputs, train_targets = augment_match_data(
        matches,
        encoder,
        augmentation_factor=augmentation_factor
    )
    
    print(f"\nTraining samples (after augmentation): {len(train_inputs)}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"Target shape: {train_targets.shape}")
    
    # Split into train and test
    test_split = 0.2
    split_idx = int(len(train_inputs) * (1 - test_split))
    
    # Shuffle data
    indices = np.random.permutation(len(train_inputs))
    train_inputs = train_inputs[indices]
    train_targets = train_targets[indices]
    
    test_inputs = train_inputs[split_idx:]
    test_targets = train_targets[split_idx:]
    train_inputs = train_inputs[:split_idx]
    train_targets = train_targets[:split_idx]
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Test samples: {len(test_inputs)}")
    
    # Determine vocabulary sizes from data
    input_vocab_size = int(np.max(train_inputs)) + 10  # Add buffer
    target_vocab_size = int(np.max(train_targets)) + 10  # Add buffer
    max_position_encoding = max(train_inputs.shape[1], train_targets.shape[1]) + 10
    
    print(f"\nInput vocab size: {input_vocab_size}")
    print(f"Target vocab size: {target_vocab_size}")
    print(f"Max position encoding: {max_position_encoding}")
    
    # Create model
    print("\nCreating transformer model...")
    model = create_tactics_transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_position_encoding=max_position_encoding,
        dropout_rate=dropout_rate
    )
    
    # Custom learning rate schedule
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    # Prepare data for training
    # Shift target sequences for teacher forcing
    train_targets_input = train_targets[:, :-1]
    train_targets_output = train_targets[:, 1:]
    
    test_targets_input = test_targets[:, :-1]
    test_targets_output = test_targets[:, 1:]
    
    # Create callbacks
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        'tactics_transformer_match_data_{epoch:02d}_{val_loss:.4f}.h5'
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\nTraining model on real match data...")
    history = model.fit(
        (train_inputs, train_targets_input),
        train_targets_output,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            (test_inputs, test_targets_input),
            test_targets_output
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'tactics_transformer_match_data_final.weights.h5')
    model.save_weights(final_model_path)
    print(f"\nModel weights saved to {final_model_path}")
    
    # Save model configuration
    config = {
        'num_layers': num_layers,
        'd_model': d_model,
        'num_heads': num_heads,
        'dff': dff,
        'input_vocab_size': input_vocab_size,
        'target_vocab_size': target_vocab_size,
        'max_position_encoding': max_position_encoding,
        'dropout_rate': dropout_rate,
    }
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model configuration saved to {config_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'masked_accuracy': [float(x) for x in history.history['masked_accuracy']],
            'val_masked_accuracy': [float(x) for x in history.history['val_masked_accuracy']],
        }, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return model, history, encoder


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model on real match data
    model, history, encoder = train_model_on_matches(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        dropout_rate=0.1,
        epochs=100,
        batch_size=16,
        save_dir='models',
        augmentation_factor=20
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history.history['masked_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_masked_accuracy'][-1]:.4f}")
    print("="*60)
