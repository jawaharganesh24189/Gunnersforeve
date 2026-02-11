"""
Data schema for Arsenal FC match datasets.

This module defines the structure for match data to ensure consistency
across different data sources.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PlayerStats(BaseModel):
    """Player statistics for a match."""
    player_name: str
    minutes_played: int
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    position: Optional[str] = None
    
    
class TeamStats(BaseModel):
    """Team statistics for a match."""
    team_name: str
    goals: int
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    possession: Optional[float] = None  # percentage
    corners: Optional[int] = None
    fouls: Optional[int] = None
    yellow_cards: Optional[int] = 0
    red_cards: Optional[int] = 0
    xg: Optional[float] = None  # Expected goals
    

class MatchData(BaseModel):
    """Complete match data structure."""
    match_id: Optional[str] = None
    date: str  # ISO format YYYY-MM-DD
    time: Optional[str] = None  # HH:MM format
    competition: str  # e.g., "Premier League", "Champions League"
    season: str  # e.g., "2023-24"
    
    # Teams
    home_team: str
    away_team: str
    is_arsenal_home: bool
    
    # Score
    home_score: int
    away_score: int
    halftime_home_score: Optional[int] = None
    halftime_away_score: Optional[int] = None
    
    # Venue
    venue: Optional[str] = None
    attendance: Optional[int] = None
    
    # Statistics
    home_stats: Optional[TeamStats] = None
    away_stats: Optional[TeamStats] = None
    
    # Players
    arsenal_lineup: Optional[List[str]] = None
    opponent_lineup: Optional[List[str]] = None
    arsenal_players: Optional[List[PlayerStats]] = None
    opponent_players: Optional[List[PlayerStats]] = None
    
    # Additional metadata
    referee: Optional[str] = None
    weather: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "PL_2023_Arsenal_vs_Chelsea",
                "date": "2023-10-21",
                "time": "15:00",
                "competition": "Premier League",
                "season": "2023-24",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "is_arsenal_home": True,
                "home_score": 2,
                "away_score": 0,
                "halftime_home_score": 1,
                "halftime_away_score": 0,
                "venue": "Emirates Stadium",
                "attendance": 60000,
                "home_stats": {
                    "team_name": "Arsenal",
                    "goals": 2,
                    "shots": 15,
                    "shots_on_target": 8,
                    "possession": 62.5,
                    "corners": 7,
                    "fouls": 9,
                    "yellow_cards": 1,
                    "red_cards": 0,
                    "xg": 2.3
                },
                "away_stats": {
                    "team_name": "Chelsea",
                    "goals": 0,
                    "shots": 8,
                    "shots_on_target": 3,
                    "possession": 37.5,
                    "corners": 3,
                    "fouls": 12,
                    "yellow_cards": 2,
                    "red_cards": 0,
                    "xg": 0.8
                }
            }
        }


class Season(BaseModel):
    """Season information."""
    season_id: str  # e.g., "2023-24"
    start_year: int
    end_year: int
    competition: str
    

class Dataset(BaseModel):
    """Collection of matches forming a dataset."""
    dataset_name: str
    description: str
    source: str
    last_updated: str  # ISO format datetime
    matches: List[MatchData]
    
    def to_csv(self, filepath: str):
        """Export dataset to CSV format."""
        import pandas as pd
        
        # Flatten match data for CSV export
        records = []
        for match in self.matches:
            record = {
                'match_id': match.match_id,
                'date': match.date,
                'time': match.time,
                'competition': match.competition,
                'season': match.season,
                'home_team': match.home_team,
                'away_team': match.away_team,
                'is_arsenal_home': match.is_arsenal_home,
                'home_score': match.home_score,
                'away_score': match.away_score,
                'halftime_home_score': match.halftime_home_score,
                'halftime_away_score': match.halftime_away_score,
                'venue': match.venue,
                'attendance': match.attendance,
            }
            
            # Add home team stats
            if match.home_stats:
                record.update({
                    'home_shots': match.home_stats.shots,
                    'home_shots_on_target': match.home_stats.shots_on_target,
                    'home_possession': match.home_stats.possession,
                    'home_corners': match.home_stats.corners,
                    'home_fouls': match.home_stats.fouls,
                    'home_yellow_cards': match.home_stats.yellow_cards,
                    'home_red_cards': match.home_stats.red_cards,
                    'home_xg': match.home_stats.xg,
                })
            
            # Add away team stats
            if match.away_stats:
                record.update({
                    'away_shots': match.away_stats.shots,
                    'away_shots_on_target': match.away_stats.shots_on_target,
                    'away_possession': match.away_stats.possession,
                    'away_corners': match.away_stats.corners,
                    'away_fouls': match.away_stats.fouls,
                    'away_yellow_cards': match.away_stats.yellow_cards,
                    'away_red_cards': match.away_stats.red_cards,
                    'away_xg': match.away_stats.xg,
                })
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        return df
