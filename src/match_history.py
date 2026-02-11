"""
Match history module for football tactics transformer.

This module handles real match data and outcomes for training the model.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import numpy as np


@dataclass
class MatchData:
    """
    Data structure for a complete match with outcomes.
    
    Stores all tactical information and actual match results.
    """
    
    # Match metadata
    match_id: str
    date: datetime
    home_team: str
    away_team: str
    
    # Match outcome
    home_goals: int
    away_goals: int
    home_possession: float  # Percentage (0-100)
    away_possession: float  # Percentage (0-100)
    
    # Advanced statistics
    home_shots: int
    away_shots: int
    home_shots_on_target: int
    away_shots_on_target: int
    home_xg: float  # Expected goals
    away_xg: float  # Expected goals
    
    # Tactical setup
    home_formation: str
    away_formation: str
    tactical_context: str
    
    # Passing sequences (list of successful passing sequences)
    # Format: List of (position, action, success_rate) tuples
    passing_sequences: Optional[List[List[Tuple[str, str, float]]]] = None
    
    def __post_init__(self):
        """Validate match data"""
        if self.home_possession + self.away_possession > 100.1:  # Allow small float error
            raise ValueError("Total possession cannot exceed 100%")
        
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("Goals cannot be negative")
    
    @property
    def winner(self) -> Optional[str]:
        """Return winning team or None for draw"""
        if self.home_goals > self.away_goals:
            return self.home_team
        elif self.away_goals > self.home_goals:
            return self.away_team
        return None
    
    @property
    def total_goals(self) -> int:
        """Return total goals in match"""
        return self.home_goals + self.away_goals
    
    def is_high_scoring(self, threshold: int = 3) -> bool:
        """Check if match was high-scoring"""
        return self.total_goals >= threshold


class MatchDataLoader:
    """
    Loads and manages match history data for training.
    """
    
    def __init__(self):
        self.matches: List[MatchData] = []
    
    def add_match(self, match: MatchData):
        """Add a match to the dataset"""
        self.matches.append(match)
    
    def get_matches_by_team(self, team_name: str) -> List[MatchData]:
        """Get all matches involving a specific team"""
        return [m for m in self.matches 
                if m.home_team == team_name or m.away_team == team_name]
    
    def get_matches_by_formation(self, formation: str) -> List[MatchData]:
        """Get matches where a team used a specific formation"""
        return [m for m in self.matches 
                if m.home_formation == formation or m.away_formation == formation]
    
    def get_high_scoring_matches(self, threshold: int = 3) -> List[MatchData]:
        """Get matches with total goals >= threshold"""
        return [m for m in self.matches if m.total_goals >= threshold]
    
    def get_possession_dominant_matches(self, threshold: float = 60.0) -> List[MatchData]:
        """Get matches where a team had >= threshold% possession"""
        return [m for m in self.matches 
                if m.home_possession >= threshold or m.away_possession >= threshold]
    
    def get_training_samples(self) -> List[Tuple[Dict, List]]:
        """
        Convert match data to training samples.
        
        Returns:
            List of (tactical_situation, passing_sequence) tuples
        """
        samples = []
        
        for match in self.matches:
            if match.passing_sequences is None:
                continue
            
            for sequence in match.passing_sequences:
                # Create tactical situation dictionary
                situation = {
                    'own_formation': match.home_formation,
                    'opponent_formation': match.away_formation,
                    'tactical_context': match.tactical_context,
                    'team': match.home_team,
                    'opponent': match.away_team,
                }
                
                samples.append((situation, sequence))
        
        return samples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.matches:
            return {}
        
        return {
            'total_matches': len(self.matches),
            'avg_goals': np.mean([m.total_goals for m in self.matches]),
            'avg_possession_home': np.mean([m.home_possession for m in self.matches]),
            'avg_shots': np.mean([m.home_shots + m.away_shots for m in self.matches]),
            'formations': list(set([m.home_formation for m in self.matches] + 
                                  [m.away_formation for m in self.matches])),
        }


def create_sample_match_data() -> List[MatchData]:
    """
    Create sample match data for demonstration.
    
    Returns:
        List of sample MatchData objects
    """
    sample_matches = [
        MatchData(
            match_id="PL_2024_001",
            date=datetime(2024, 1, 15),
            home_team="Arsenal",
            away_team="Manchester City",
            home_goals=3,
            away_goals=1,
            home_possession=48.0,
            away_possession=52.0,
            home_shots=15,
            away_shots=12,
            home_shots_on_target=8,
            away_shots_on_target=5,
            home_xg=2.4,
            away_xg=1.1,
            home_formation="4-3-3",
            away_formation="4-3-3",
            tactical_context="counter_attack",
            passing_sequences=[
                [('CB', 'short_pass', 0.92), ('CDM', 'forward_pass', 0.88), ('CAM', 'through_ball', 0.75), ('ST', 'shot', 0.65)],
                [('GK', 'long_pass', 0.70), ('ST', 'header', 0.55), ('CAM', 'shot', 0.60)],
            ]
        ),
        MatchData(
            match_id="SA_2024_001",
            date=datetime(2024, 1, 20),
            home_team="Napoli",
            away_team="Inter Milan",
            home_goals=2,
            away_goals=2,
            home_possession=55.0,
            away_possession=45.0,
            home_shots=18,
            away_shots=10,
            home_shots_on_target=7,
            away_shots_on_target=6,
            home_xg=1.8,
            away_xg=1.9,
            home_formation="4-3-3",
            away_formation="3-5-2",
            tactical_context="possession",
            passing_sequences=[
                [('CB', 'short_pass', 0.95), ('CM', 'short_pass', 0.93), ('CAM', 'through_ball', 0.78), ('ST', 'shot', 0.62)],
            ]
        ),
        MatchData(
            match_id="L1_2024_001",
            date=datetime(2024, 1, 25),
            home_team="Paris Saint-Germain",
            away_team="Marseille",
            home_goals=4,
            away_goals=0,
            home_possession=62.0,
            away_possession=38.0,
            home_shots=22,
            away_shots=6,
            home_shots_on_target=12,
            away_shots_on_target=2,
            home_xg=3.5,
            away_xg=0.4,
            home_formation="4-3-3",
            away_formation="3-4-3",
            tactical_context="high_press",
            passing_sequences=[
                [('CDM', 'short_pass', 0.94), ('LW', 'forward_pass', 0.85), ('ST', 'shot', 0.70)],
                [('CB', 'long_pass', 0.82), ('RW', 'cross', 0.75), ('ST', 'header', 0.68)],
            ]
        ),
        MatchData(
            match_id="LL_2024_001",
            date=datetime(2024, 2, 1),
            home_team="Real Madrid",
            away_team="Barcelona",
            home_goals=2,
            away_goals=3,
            home_possession=45.0,
            away_possession=55.0,
            home_shots=11,
            away_shots=16,
            home_shots_on_target=6,
            away_shots_on_target=9,
            home_xg=1.6,
            away_xg=2.7,
            home_formation="4-3-3",
            away_formation="4-3-3",
            tactical_context="possession",
            passing_sequences=[
                [('CB', 'short_pass', 0.96), ('CM', 'short_pass', 0.94), ('CM', 'forward_pass', 0.89), ('CAM', 'through_ball', 0.80), ('ST', 'shot', 0.68)],
            ]
        ),
        MatchData(
            match_id="BL_2024_001",
            date=datetime(2024, 2, 5),
            home_team="Bayern Munich",
            away_team="Borussia Dortmund",
            home_goals=3,
            away_goals=2,
            home_possession=58.0,
            away_possession=42.0,
            home_shots=19,
            away_shots=13,
            home_shots_on_target=10,
            away_shots_on_target=7,
            home_xg=2.8,
            away_xg=1.9,
            home_formation="4-2-3-1",
            away_formation="4-3-3",
            tactical_context="build_from_back",
            passing_sequences=[
                [('CB', 'short_pass', 0.93), ('CDM', 'forward_pass', 0.87), ('CAM', 'through_ball', 0.76), ('ST', 'shot', 0.71)],
            ]
        ),
        # Additional matches
        MatchData(
            match_id="PL_2024_002",
            date=datetime(2024, 2, 10),
            home_team="Liverpool",
            away_team="Chelsea",
            home_goals=4,
            away_goals=1,
            home_possession=60.0,
            away_possession=40.0,
            home_shots=20,
            away_shots=8,
            home_shots_on_target=11,
            away_shots_on_target=3,
            home_xg=3.2,
            away_xg=0.8,
            home_formation="4-3-3",
            away_formation="3-4-3",
            tactical_context="high_press",
            passing_sequences=[
                [('CB', 'short_pass', 0.91), ('LB', 'forward_pass', 0.86), ('LW', 'through_ball', 0.79), ('ST', 'shot', 0.72)],
                [('CDM', 'long_pass', 0.78), ('RW', 'cross', 0.73), ('ST', 'header', 0.67)],
            ]
        ),
        MatchData(
            match_id="PL_2024_003",
            date=datetime(2024, 2, 15),
            home_team="Manchester United",
            away_team="Tottenham",
            home_goals=2,
            away_goals=2,
            home_possession=52.0,
            away_possession=48.0,
            home_shots=14,
            away_shots=13,
            home_shots_on_target=6,
            away_shots_on_target=7,
            home_xg=1.9,
            away_xg=2.0,
            home_formation="4-2-3-1",
            away_formation="4-2-3-1",
            tactical_context="counter_attack",
            passing_sequences=[
                [('CB', 'short_pass', 0.88), ('CM', 'forward_pass', 0.84), ('CAM', 'through_ball', 0.74), ('ST', 'shot', 0.66)],
            ]
        ),
        MatchData(
            match_id="SA_2024_002",
            date=datetime(2024, 2, 20),
            home_team="AC Milan",
            away_team="Juventus",
            home_goals=1,
            away_goals=0,
            home_possession=49.0,
            away_possession=51.0,
            home_shots=12,
            away_shots=14,
            home_shots_on_target=5,
            away_shots_on_target=4,
            home_xg=1.3,
            away_xg=1.2,
            home_formation="4-2-3-1",
            away_formation="3-5-2",
            tactical_context="low_block",
            passing_sequences=[
                [('CB', 'long_pass', 0.75), ('ST', 'control', 0.68), ('CAM', 'through_ball', 0.71), ('ST', 'shot', 0.64)],
            ]
        ),
        MatchData(
            match_id="SA_2024_003",
            date=datetime(2024, 2, 25),
            home_team="Atalanta",
            away_team="Roma",
            home_goals=3,
            away_goals=1,
            home_possession=54.0,
            away_possession=46.0,
            home_shots=17,
            away_shots=11,
            home_shots_on_target=9,
            away_shots_on_target=4,
            home_xg=2.6,
            away_xg=1.1,
            home_formation="3-4-3",
            away_formation="3-4-2-1",
            tactical_context="high_press",
            passing_sequences=[
                [('CB', 'short_pass', 0.90), ('RWB', 'forward_pass', 0.85), ('RW', 'cross', 0.77), ('ST', 'header', 0.69)],
                [('CM', 'through_ball', 0.80), ('LW', 'shot', 0.70)],
            ]
        ),
        MatchData(
            match_id="L1_2024_002",
            date=datetime(2024, 3, 1),
            home_team="Monaco",
            away_team="Lyon",
            home_goals=2,
            away_goals=1,
            home_possession=50.0,
            away_possession=50.0,
            home_shots=13,
            away_shots=12,
            home_shots_on_target=6,
            away_shots_on_target=5,
            home_xg=1.7,
            away_xg=1.4,
            home_formation="4-4-2",
            away_formation="4-3-3",
            tactical_context="direct_play",
            passing_sequences=[
                [('CB', 'long_pass', 0.76), ('ST', 'control', 0.72), ('ST', 'shot', 0.65)],
            ]
        ),
        MatchData(
            match_id="L1_2024_003",
            date=datetime(2024, 3, 5),
            home_team="Lille",
            away_team="Rennes",
            home_goals=1,
            away_goals=1,
            home_possession=47.0,
            away_possession=53.0,
            home_shots=10,
            away_shots=14,
            home_shots_on_target=4,
            away_shots_on_target=6,
            home_xg=1.1,
            away_xg=1.5,
            home_formation="4-2-3-1",
            away_formation="4-3-3",
            tactical_context="possession",
            passing_sequences=[
                [('CB', 'short_pass', 0.92), ('CM', 'short_pass', 0.90), ('CAM', 'through_ball', 0.76), ('ST', 'shot', 0.63)],
            ]
        ),
        MatchData(
            match_id="LL_2024_002",
            date=datetime(2024, 3, 10),
            home_team="Atletico Madrid",
            away_team="Sevilla",
            home_goals=1,
            away_goals=0,
            home_possession=42.0,
            away_possession=58.0,
            home_shots=8,
            away_shots=16,
            home_shots_on_target=3,
            away_shots_on_target=5,
            home_xg=0.9,
            away_xg=1.6,
            home_formation="3-5-2",
            away_formation="4-3-3",
            tactical_context="low_block",
            passing_sequences=[
                [('CB', 'long_pass', 0.73), ('ST', 'control', 0.70), ('ST', 'shot', 0.68)],
            ]
        ),
        MatchData(
            match_id="LL_2024_003",
            date=datetime(2024, 3, 15),
            home_team="Real Sociedad",
            away_team="Real Betis",
            home_goals=2,
            away_goals=2,
            home_possession=56.0,
            away_possession=44.0,
            home_shots=15,
            away_shots=11,
            home_shots_on_target=7,
            away_shots_on_target=6,
            home_xg=2.1,
            away_xg=1.8,
            home_formation="4-2-3-1",
            away_formation="4-2-3-1",
            tactical_context="possession",
            passing_sequences=[
                [('CB', 'short_pass', 0.93), ('CDM', 'forward_pass', 0.88), ('CAM', 'through_ball', 0.77), ('ST', 'shot', 0.70)],
            ]
        ),
        MatchData(
            match_id="BL_2024_002",
            date=datetime(2024, 3, 20),
            home_team="RB Leipzig",
            away_team="Bayer Leverkusen",
            home_goals=3,
            away_goals=2,
            home_possession=53.0,
            away_possession=47.0,
            home_shots=16,
            away_shots=13,
            home_shots_on_target=8,
            away_shots_on_target=7,
            home_xg=2.4,
            away_xg=2.0,
            home_formation="3-4-3",
            away_formation="4-2-3-1",
            tactical_context="high_press",
            passing_sequences=[
                [('CB', 'short_pass', 0.89), ('CM', 'forward_pass', 0.86), ('CAM', 'through_ball', 0.78), ('ST', 'shot', 0.72)],
                [('RWB', 'cross', 0.74), ('ST', 'header', 0.68)],
            ]
        ),
        MatchData(
            match_id="BL_2024_003",
            date=datetime(2024, 3, 25),
            home_team="Union Berlin",
            away_team="Eintracht Frankfurt",
            home_goals=1,
            away_goals=1,
            home_possession=44.0,
            away_possession=56.0,
            home_shots=9,
            away_shots=14,
            home_shots_on_target=4,
            away_shots_on_target=6,
            home_xg=1.0,
            away_xg=1.4,
            home_formation="3-5-2",
            away_formation="3-4-2-1",
            tactical_context="counter_attack",
            passing_sequences=[
                [('CB', 'long_pass', 0.72), ('ST', 'control', 0.68), ('ST', 'shot', 0.65)],
            ]
        ),
    ]
    
    return sample_matches


def load_match_history() -> MatchDataLoader:
    """
    Load sample match history data.
    
    Returns:
        MatchDataLoader with sample matches
    """
    loader = MatchDataLoader()
    for match in create_sample_match_data():
        loader.add_match(match)
    return loader
