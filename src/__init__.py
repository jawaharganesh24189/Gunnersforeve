"""
Gunnersforeve - Football Tactics Transformer

A Keras-based transformer model for generating passing tactics.
Improved with DLA (Deep Learning Architecture) best practices.
"""

from .transformer_model import (
    TacticsTransformer,
    create_tactics_transformer,
    PositionalEncoding,
    FeedForward,
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

__version__ = '1.1.0'  # Updated version for DLA improvements
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
    'FeedForward',
    'EncoderLayer',
    'DecoderLayer'
]
