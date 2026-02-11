"""
Gunnersforeve - Arsenal FC Match Data Collection and Analysis

This package provides tools for collecting and analyzing Arsenal FC match data.
"""

__version__ = "1.0.0"
__author__ = "Arsenal Data Team"

from .data_schema import MatchData, TeamStats, PlayerStats, Dataset, Season
from .data_collector import FootballDataCollector, create_dataset_from_api, save_dataset
from .simulator import FootballMatchSimulator, create_simulated_dataset, TeamProfile

__all__ = [
    "MatchData",
    "TeamStats",
    "PlayerStats",
    "Dataset",
    "Season",
    "FootballDataCollector",
    "create_dataset_from_api",
    "save_dataset",
    "FootballMatchSimulator",
    "create_simulated_dataset",
    "TeamProfile",
]
