"""
Transformer Model for Football Passing Tactics Generation

This module implements a Keras-based transformer model that can generate
passing tactics from the backline to the opposite goal, considering different
oppositions, formations, and tactical situations.

Improvements based on DLA (Deep Learning Architecture) best practices:
- Uses Keras built-in MultiHeadAttention for optimized performance
- Implements Pre-LayerNorm architecture for better training stability
- Supports both fixed and learnable positional embeddings
- Includes proper type hints for better maintainability
- Improved attention masking and gradient handling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(layers.Layer):
    """
    Implements positional encoding for the transformer model.
    This helps the model understand the sequence order of passes.
    
    Supports both fixed sinusoidal encoding (original Transformer)
    and learnable position embeddings (more flexible).
    """
    
    def __init__(self, max_position: int, d_model: int, learnable: bool = False):
        """
        Args:
            max_position: Maximum sequence length
            d_model: Dimension of the model
            learnable: If True, uses learnable embeddings; if False, uses fixed sinusoidal
        """
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
        self.learnable = learnable
        
        if learnable:
            # Learnable positional embeddings (modern approach)
            self.pos_embedding = layers.Embedding(
                input_dim=max_position,
                output_dim=d_model
            )
        else:
            # Fixed sinusoidal encoding (original Transformer)
            self.pos_encoding = self._positional_encoding(max_position, d_model)
    
    def _positional_encoding(self, max_position: int, d_model: int) -> tf.Tensor:
        """Generate fixed sinusoidal positional encoding matrix"""
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_position, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to input embeddings"""
        seq_length = tf.shape(inputs)[1]
        
        if self.learnable:
            # Use learnable position embeddings
            positions = tf.range(start=0, limit=seq_length, delta=1)
            pos_emb = self.pos_embedding(positions)
            return inputs + pos_emb
        else:
            # Use fixed sinusoidal encoding
            return inputs + self.pos_encoding[:, :seq_length, :]



class FeedForward(layers.Layer):
    """
    Position-wise feed-forward network with GELU activation (modern standard).
    """
    
    def __init__(self, d_model: int, dff: int):
        """
        Args:
            d_model: Model dimension
            dff: Feed-forward dimension (typically 4 * d_model)
        """
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(dff, activation='gelu')  # GELU instead of ReLU
        self.dense2 = layers.Dense(d_model)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply feed-forward transformation"""
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EncoderLayer(layers.Layer):
    """
    Single encoder layer using Pre-LayerNorm architecture.
    
    Pre-LN (LayerNorm before attention/FFN) provides:
    - Better gradient flow for deeper models
    - More stable training
    - Better performance than Post-LN architecture
    """
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward dimension
            dropout_rate: Dropout rate for regularization
        """
        super(EncoderLayer, self).__init__()
        
        # Use Keras built-in MultiHeadAttention (optimized)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)
        
        # Pre-LN: LayerNorm before each sub-layer
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """
        Forward pass with Pre-LayerNorm architecture.
        
        Args:
            x: Input tensor
            mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Transformed tensor
        """
        # Pre-LN: LayerNorm before attention
        attn_input = self.layernorm1(x)
        attn_output = self.mha(
            query=attn_input,
            value=attn_input,
            key=attn_input,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output  # Residual connection
        
        # Pre-LN: LayerNorm before FFN
        ffn_input = self.layernorm2(out1)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output  # Residual connection
        
        return out2


class DecoderLayer(layers.Layer):
    """
    Single decoder layer with Pre-LayerNorm architecture.
    
    Includes:
    - Masked self-attention (prevents looking ahead)
    - Cross-attention to encoder output
    - Feed-forward network
    """
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        # Use Keras built-in MultiHeadAttention
        self.mha1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.mha2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)
        
        # Pre-LN architecture
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        look_ahead_mask: Optional[tf.Tensor] = None,
        padding_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass with Pre-LayerNorm architecture.
        
        Args:
            x: Decoder input
            enc_output: Encoder output
            look_ahead_mask: Mask for decoder self-attention
            padding_mask: Mask for encoder-decoder attention
            training: Whether in training mode
            
        Returns:
            Transformed tensor
        """
        # Pre-LN: Masked self-attention
        attn1_input = self.layernorm1(x)
        attn1 = self.mha1(
            query=attn1_input,
            value=attn1_input,
            key=attn1_input,
            attention_mask=look_ahead_mask,
            training=training
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1
        
        # Pre-LN: Cross-attention with encoder output
        attn2_input = self.layernorm2(out1)
        attn2 = self.mha2(
            query=attn2_input,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = out1 + attn2
        
        # Pre-LN: Feed-forward network
        ffn_input = self.layernorm3(out2)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = out2 + ffn_output
        
        return out3


class TacticsTransformer(keras.Model):
    """
    Complete Transformer model for generating passing tactics.
    
    Implements modern best practices:
    - Pre-LayerNorm architecture for stability
    - Keras built-in attention for performance
    - Proper attention masking
    - Support for learnable positional embeddings
    
    The model takes as input:
    - Formation data (both team and opposition)
    - Player positions
    - Current ball position
    - Tactical context
    
    And generates:
    - Sequence of passes from backline to opposite goal
    - Player positions for each pass
    - Tactical instructions
    """
    
    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        dff: int = 512,
        input_vocab_size: int = 1000,
        target_vocab_size: int = 1000,
        max_position_encoding: int = 100,
        dropout_rate: float = 0.1,
        learnable_pos_encoding: bool = False,
        pad_token_id: int = 0
    ):
        """
        Args:
            num_layers: Number of encoder/decoder layers
            d_model: Model embedding dimension
            num_heads: Number of attention heads
            dff: Feed-forward network dimension
            input_vocab_size: Size of input vocabulary
            target_vocab_size: Size of target vocabulary
            max_position_encoding: Maximum sequence length
            dropout_rate: Dropout rate for regularization
            learnable_pos_encoding: Use learnable vs fixed positional encoding
            pad_token_id: Token ID used for padding (default: 0)
        """
        super(TacticsTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Embedding layers
        self.embedding_input = layers.Embedding(input_vocab_size, d_model)
        self.embedding_target = layers.Embedding(target_vocab_size, d_model)
        
        # Positional encoding (learnable or fixed)
        self.pos_encoding_input = PositionalEncoding(
            max_position_encoding, d_model, learnable=learnable_pos_encoding
        )
        self.pos_encoding_target = PositionalEncoding(
            max_position_encoding, d_model, learnable=learnable_pos_encoding
        )
        
        # Encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
        
        # Final layer normalization (Pre-LN architecture)
        self.final_layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        # Final output layer
        self.final_layer = layers.Dense(target_vocab_size)
    
    def create_look_ahead_mask(self, size: int) -> tf.Tensor:
        """
        Creates causal (look-ahead) mask for decoder to prevent attending to future tokens.
        
        Args:
            size: Sequence length
            
        Returns:
            Boolean mask of shape (size, size)
        """
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # 1 for positions to keep, 0 for positions to mask
    
    def create_padding_mask(self, seq: tf.Tensor) -> tf.Tensor:
        """
        Creates padding mask for sequences.
        
        Args:
            seq: Input sequence tensor
            
        Returns:
            Boolean mask where True indicates positions to keep
        """
        # Keras MultiHeadAttention expects True for positions to keep
        mask = tf.cast(tf.math.not_equal(seq, self.pad_token_id), tf.float32)
        # Shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len) for broadcasting
        return mask[:, tf.newaxis, tf.newaxis, :]
    
    def encode(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Encoder forward pass.
        
        Args:
            inputs: Input token IDs
            mask: Padding mask
            training: Whether in training mode
            
        Returns:
            Encoded representations
        """
        # Embedding and positional encoding
        x = self.embedding_input(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scale embeddings
        x = self.pos_encoding_input(x)
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask=mask, training=training)
        
        return x
    
    def decode(
        self,
        targets: tf.Tensor,
        enc_output: tf.Tensor,
        look_ahead_mask: Optional[tf.Tensor] = None,
        padding_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Decoder forward pass.
        
        Args:
            targets: Target token IDs
            enc_output: Encoder output
            look_ahead_mask: Causal mask for decoder self-attention
            padding_mask: Padding mask for encoder-decoder attention
            training: Whether in training mode
            
        Returns:
            Decoded representations
        """
        # Embedding and positional encoding
        x = self.embedding_target(targets)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scale embeddings
        x = self.pos_encoding_target(x)
        x = self.dropout(x, training=training)
        
        # Pass through decoder layers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](
                x, enc_output, look_ahead_mask=look_ahead_mask, 
                padding_mask=padding_mask, training=training
            )
        
        # Final layer normalization (Pre-LN architecture)
        x = self.final_layernorm(x)
        
        return x
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            inputs: Tuple of (encoder_inputs, decoder_inputs)
            training: Boolean indicating training mode
        
        Returns:
            Model predictions of shape (batch_size, target_seq_len, target_vocab_size)
        """
        inp, tar = inputs
        
        # Create masks
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Create look-ahead mask for decoder
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        
        # Combine look-ahead and padding masks
        # Use minimum to combine (0 = mask, 1 = keep)
        combined_mask = tf.minimum(dec_target_padding_mask, look_ahead_mask)
        
        # Encode
        enc_output = self.encode(inp, mask=enc_padding_mask, training=training)
        
        # Decode
        dec_output = self.decode(
            tar, enc_output, look_ahead_mask=combined_mask, 
            padding_mask=dec_padding_mask, training=training
        )
        
        # Final linear layer
        final_output = self.final_layer(dec_output)
        
        return final_output


def create_tactics_transformer(
    num_layers: int = 4,
    d_model: int = 256,
    num_heads: int = 8,
    dff: int = 512,
    input_vocab_size: int = 1000,
    target_vocab_size: int = 1000,
    max_position_encoding: int = 100,
    dropout_rate: float = 0.1,
    learnable_pos_encoding: bool = False,
    pad_token_id: int = 0
) -> TacticsTransformer:
    """
    Factory function to create a TacticsTransformer model with modern DLA best practices.
    
    Improvements over original implementation:
    - Uses Keras built-in MultiHeadAttention (optimized)
    - Implements Pre-LayerNorm architecture (better stability)
    - Supports learnable positional embeddings
    - Uses GELU activation (modern standard)
    - Proper attention masking
    - Type hints for better maintainability
    
    Args:
        num_layers: Number of encoder/decoder layers
        d_model: Dimension of model embeddings (must be divisible by num_heads)
        num_heads: Number of attention heads
        dff: Dimension of feed-forward network (typically 4 * d_model)
        input_vocab_size: Size of input vocabulary (formations, positions, etc.)
        target_vocab_size: Size of output vocabulary (passing actions)
        max_position_encoding: Maximum sequence length
        dropout_rate: Dropout rate for regularization
        learnable_pos_encoding: Use learnable vs fixed sinusoidal positional encoding
        pad_token_id: Token ID used for padding (ensure this doesn't conflict with vocab)
    
    Returns:
        Compiled TacticsTransformer model with improved architecture
        
    Example:
        >>> model = create_tactics_transformer(
        ...     num_layers=6,
        ...     d_model=512,
        ...     num_heads=8,
        ...     dff=2048,
        ...     learnable_pos_encoding=True
        ... )
    """
    # Validate parameters
    assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
    
    model = TacticsTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_position_encoding=max_position_encoding,
        dropout_rate=dropout_rate,
        learnable_pos_encoding=learnable_pos_encoding,
        pad_token_id=pad_token_id
    )
    
    return model
