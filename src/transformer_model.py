"""
Transformer Model for Football Passing Tactics Generation

This module implements a Keras-based transformer model that can generate
passing tactics from the backline to the opposite goal, considering different
oppositions, formations, and tactical situations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    """
    Implements positional encoding for the transformer model.
    This helps the model understand the sequence order of passes.
    """
    
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(max_position, d_model)
    
    def _positional_encoding(self, max_position, d_model):
        """Generate positional encoding matrix"""
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_position, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to input embeddings"""
        length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :length, :]


class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention mechanism for the transformer.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        
        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(output)
        return output


class FeedForward(layers.Layer):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EncoderLayer(layers.Layer):
    """
    Single encoder layer consisting of multi-head attention and feed-forward network.
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, mask=None, training=False):
        # Multi-head attention
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class DecoderLayer(layers.Layer):
    """
    Single decoder layer with masked multi-head attention, encoder-decoder attention,
    and feed-forward network.
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        # Masked multi-head attention (self-attention)
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Multi-head attention with encoder output
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3


class TacticsTransformer(keras.Model):
    """
    Complete Transformer model for generating passing tactics.
    
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
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        input_vocab_size=1000,
        target_vocab_size=1000,
        max_position_encoding=100,
        dropout_rate=0.1
    ):
        super(TacticsTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layers
        self.embedding_input = layers.Embedding(input_vocab_size, d_model)
        self.embedding_target = layers.Embedding(target_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding_input = PositionalEncoding(max_position_encoding, d_model)
        self.pos_encoding_target = PositionalEncoding(max_position_encoding, d_model)
        
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
        
        # Final output layer
        self.final_layer = layers.Dense(target_vocab_size)
    
    def create_look_ahead_mask(self, size):
        """Creates look-ahead mask for decoder to prevent attending to future tokens"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def create_padding_mask(self, seq):
        """Creates padding mask for sequences"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def encode(self, inputs, mask=None, training=False):
        """Encoder forward pass"""
        # Embedding and positional encoding
        x = self.embedding_input(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding_input(x)
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask=mask, training=training)
        
        return x
    
    def decode(self, targets, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        """Decoder forward pass"""
        # Embedding and positional encoding
        x = self.embedding_target(targets)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding_target(x)
        x = self.dropout(x, training=training)
        
        # Pass through decoder layers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](
                x, enc_output, look_ahead_mask=look_ahead_mask, 
                padding_mask=padding_mask, training=training
            )
        
        return x
    
    def call(self, inputs, training=False):
        """
        Forward pass of the transformer.
        
        Args:
            inputs: Tuple of (encoder_inputs, decoder_inputs)
            training: Boolean indicating training mode
        
        Returns:
            Model predictions
        """
        inp, tar = inputs
        
        # Create masks
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
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
    num_layers=4,
    d_model=256,
    num_heads=8,
    dff=512,
    input_vocab_size=1000,
    target_vocab_size=1000,
    max_position_encoding=100,
    dropout_rate=0.1
):
    """
    Factory function to create a TacticsTransformer model.
    
    Args:
        num_layers: Number of encoder/decoder layers
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        dff: Dimension of feed-forward network
        input_vocab_size: Size of input vocabulary (formations, positions, etc.)
        target_vocab_size: Size of output vocabulary (passing actions)
        max_position_encoding: Maximum sequence length
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled TacticsTransformer model
    """
    model = TacticsTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_position_encoding=max_position_encoding,
        dropout_rate=dropout_rate
    )
    
    return model
