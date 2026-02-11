"""
Training script for the Tactics Transformer model.

This script demonstrates how to train the transformer model on football tactics data.
Improvements based on DLA best practices:
- Gradient clipping for training stability
- Improved learning rate schedule with configurable warmup
- Better optimizer configuration
- Type hints for maintainability
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any

from .transformer_model import create_tactics_transformer
from .data_preprocessing import prepare_training_data, TacticsEncoder


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule for transformer training.
    Implements warmup and inverse square root decay strategy from "Attention is All You Need".
    """
    
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        """
        Args:
            d_model: Model dimension
            warmup_steps: Number of warmup steps (default: 4000 as per original paper)
        """
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step: int) -> tf.Tensor:
        """
        Calculate learning rate for given step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate value
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(real: tf.Tensor, pred: tf.Tensor, pad_token_id: int = 0) -> tf.Tensor:
    """
    Masked loss function that ignores padding tokens.
    
    Args:
        real: Ground truth labels
        pred: Model predictions (logits)
        pad_token_id: Token ID used for padding (default: 0)
        
    Returns:
        Scalar loss value
    """
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    # Return mean loss over non-padding tokens
    # Use keras.backend.epsilon() for better numerical stability
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + keras.backend.epsilon())


def masked_accuracy(real: tf.Tensor, pred: tf.Tensor, pad_token_id: int = 0) -> tf.Tensor:
    """
    Masked accuracy metric that ignores padding tokens.
    
    Args:
        real: Ground truth labels
        pred: Model predictions (logits)
        pad_token_id: Token ID used for padding (default: 0)
        
    Returns:
        Scalar accuracy value
    """
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.cast(mask, dtype=tf.float32)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    
    # Return mean accuracy over non-padding tokens
    # Use keras.backend.epsilon() for better numerical stability
    return tf.reduce_sum(accuracies * mask) / (tf.reduce_sum(mask) + keras.backend.epsilon())


def train_model(
    num_samples: int = 1000,
    num_layers: int = 4,
    d_model: int = 256,
    num_heads: int = 8,
    dff: int = 512,
    dropout_rate: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    save_dir: str = 'models',
    gradient_clip_norm: float = 1.0,
    warmup_steps: int = None,
    learnable_pos_encoding: bool = False
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Train the tactics transformer model with improved DLA best practices.
    
    Args:
        num_samples: Number of training samples to generate
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward network dimension
        dropout_rate: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save trained models
        gradient_clip_norm: Gradient clipping norm (default: 1.0)
        warmup_steps: Number of warmup steps (default: num_samples / batch_size / 10)
        learnable_pos_encoding: Use learnable positional embeddings
    
    Returns:
        Tuple of (trained model, training history)
    """
    print("=" * 70)
    print("TACTICS TRANSFORMER TRAINING")
    print("=" * 70)
    
    print("\nPreparing training data...")
    (train_inputs, train_targets), (test_inputs, test_targets) = prepare_training_data(
        num_samples=num_samples,
        test_split=0.2
    )
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Test samples: {len(test_inputs)}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"Target shape: {train_targets.shape}")
    
    # Determine vocabulary sizes from data
    input_vocab_size = int(np.max(train_inputs)) + 1
    target_vocab_size = int(np.max(train_targets)) + 1
    max_position_encoding = max(train_inputs.shape[1], train_targets.shape[1])
    
    print(f"\nVocabulary Configuration:")
    print(f"  Input vocab size: {input_vocab_size}")
    print(f"  Target vocab size: {target_vocab_size}")
    print(f"  Max position encoding: {max_position_encoding}")
    
    # Calculate warmup steps if not provided
    if warmup_steps is None:
        steps_per_epoch = len(train_inputs) // batch_size
        warmup_steps = max(steps_per_epoch * 2, 1000)  # 2 epochs or minimum 1000
    
    print(f"\nTraining Configuration:")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Gradient clip norm: {gradient_clip_norm}")
    print(f"  Learnable pos encoding: {learnable_pos_encoding}")
    
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
        dropout_rate=dropout_rate,
        learnable_pos_encoding=learnable_pos_encoding,
        pad_token_id=0
    )
    
    # Custom learning rate schedule
    learning_rate = CustomSchedule(d_model, warmup_steps=warmup_steps)
    
    # Optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        clipnorm=gradient_clip_norm  # Add gradient clipping for stability
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    print("\nModel Architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  dff: {dff}")
    print(f"  dropout: {dropout_rate}")
    
    # Prepare data for training
    # Shift target sequences for teacher forcing
    train_targets_input = train_targets[:, :-1]
    train_targets_output = train_targets[:, 1:]
    
    test_targets_input = test_targets[:, :-1]
    test_targets_output = test_targets[:, 1:]
    
    # Create callbacks
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        'tactics_transformer_{epoch:02d}_{val_loss:.4f}.h5'
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1,
            save_weights_only=True  # Save weights only for easier loading
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
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
    final_model_path = os.path.join(save_dir, 'tactics_transformer_final.h5')
    model.save_weights(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    
    return model, history


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\n" + "=" * 70)
    print("TACTICS TRANSFORMER - TRAINING WITH DLA IMPROVEMENTS")
    print("=" * 70)
    print("\nImprovements:")
    print("✓ Pre-LayerNorm architecture for better stability")
    print("✓ Keras built-in MultiHeadAttention (optimized)")
    print("✓ Gradient clipping for stable training")
    print("✓ GELU activation (modern standard)")
    print("✓ Configurable positional encoding (fixed/learnable)")
    print("✓ Improved learning rate schedule")
    print("=" * 70 + "\n")
    
    # Train model with improved configuration
    model, history = train_model(
        num_samples=1000,
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        dropout_rate=0.1,
        epochs=50,
        batch_size=32,
        save_dir='models',
        gradient_clip_norm=1.0,
        warmup_steps=None,  # Auto-calculate
        learnable_pos_encoding=False  # Set to True for learnable embeddings
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    print(f"  Training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Training accuracy: {history.history['masked_accuracy'][-1]:.4f}")
    print(f"  Validation accuracy: {history.history['val_masked_accuracy'][-1]:.4f}")
    print("\n" + "=" * 70)
