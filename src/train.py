"""
Training script for the Tactics Transformer model.

This script demonstrates how to train the transformer model on football tactics data.
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

from transformer_model import create_tactics_transformer
from data_preprocessing import prepare_training_data, TacticsEncoder


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
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.cast(mask, dtype=tf.float32)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    
    return tf.reduce_sum(accuracies * mask) / tf.reduce_sum(mask)


def train_model(
    num_samples=1000,
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512,
    dropout_rate=0.1,
    epochs=50,
    batch_size=32,
    save_dir='models'
):
    """
    Train the tactics transformer model.
    
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
    
    Returns:
        Trained model
    """
    print("Preparing training data...")
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
    
    print(f"Input vocab size: {input_vocab_size}")
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
            verbose=1
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
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\nTraining model...")
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
    
    # Train model
    model, history = train_model(
        num_samples=1000,
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        dropout_rate=0.1,
        epochs=50,
        batch_size=32,
        save_dir='models'
    )
    
    print("\nTraining complete!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history.history['masked_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_masked_accuracy'][-1]:.4f}")
