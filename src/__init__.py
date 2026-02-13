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

from .teams_data import (
    TeamAttributes,
    League,
    TEAMS_DATABASE,
    get_team_by_name,
    get_teams_by_league,
    get_all_teams,
    get_team_names
)

from .player_stats import (
    PlayerStats,
    EXAMPLE_PLAYERS,
    create_player_stats,
    get_player_by_name
)

from .match_history import (
    MatchData,
    MatchDataLoader,
    create_sample_match_data,
    load_match_history
)

from .pancake_predictor import (
    generate_market_data,
    create_sequences,
    build_pancake_model,
    trade_logic,
    run_prediction_pipeline,
    SEQ_LENGTH,
    PREDICT_AHEAD,
    FEATURES
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
    'DecoderLayer',
    'TeamAttributes',
    'League',
    'TEAMS_DATABASE',
    'get_team_by_name',
    'get_teams_by_league',
    'get_all_teams',
    'get_team_names',
    'PlayerStats',
    'EXAMPLE_PLAYERS',
    'create_player_stats',
    'get_player_by_name',
    'MatchData',
    'MatchDataLoader',
    'create_sample_match_data',
    'load_match_history',
    'generate_market_data',
    'create_sequences',
    'build_pancake_model',
    'trade_logic',
    'run_prediction_pipeline',
    'SEQ_LENGTH',
    'PREDICT_AHEAD',
    'FEATURES'
]
