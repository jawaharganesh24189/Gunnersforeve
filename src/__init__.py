"""
Gunnersforeve - Football Tactics Transformer

A Keras-based transformer model for generating passing tactics.
"""

from .transformer_model import (
    TacticsTransformer,
    create_tactics_transformer,
    PositionalEncoding,
    MultiHeadAttention,
    EncoderLayer,
    DecoderLayer
)

from .data_preprocessing import (
    TacticsEncoder,
    TacticsDataset,
    prepare_training_data
)

from .inference import (
    TacticsGenerator,
    load_model_for_inference
)

__version__ = '1.0.0'
__author__ = 'Gunnersforeve Team'

__all__ = [
    'TacticsTransformer',
    'create_tactics_transformer',
    'TacticsEncoder',
    'TacticsDataset',
    'TacticsGenerator',
    'prepare_training_data',
    'load_model_for_inference',
    'PositionalEncoding',
    'MultiHeadAttention',
    'EncoderLayer',
    'DecoderLayer'
]
