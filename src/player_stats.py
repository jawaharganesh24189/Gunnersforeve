"""
Player statistics module for football tactics transformer.

This module contains data structures for individual player ratings and attributes.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class PlayerStats:
    """
    Individual player statistics and attributes.
    
    Attributes represent key abilities on a 1-100 scale:
    - pace: Speed and acceleration
    - passing: Passing accuracy and vision
    - shooting: Finishing and shot power
    - defending: Tackling and positioning
    - physical: Strength and stamina
    """
    
    name: str
    pace: int  # 1-100
    passing: int  # 1-100
    shooting: int  # 1-100
    defending: int  # 1-100
    physical: int  # 1-100
    overall: Optional[int] = None
    
    def __post_init__(self):
        """Calculate overall rating if not provided"""
        if self.overall is None:
            self.overall = self._calculate_overall()
        
        # Validate ratings
        for attr in ['pace', 'passing', 'shooting', 'defending', 'physical']:
            value = getattr(self, attr)
            if not 1 <= value <= 100:
                raise ValueError(f"{attr} must be between 1 and 100, got {value}")
    
    def _calculate_overall(self) -> int:
        """Calculate overall rating from individual attributes"""
        return (self.pace + self.passing + self.shooting + 
                self.defending + self.physical) // 5
    
    def get_position_rating(self, position: str) -> int:
        """
        Get player rating for a specific position.
        
        Different positions weight different attributes.
        
        Args:
            position: Player position (GK, CB, LB, RB, CDM, CM, CAM, LW, RW, ST, etc.)
        
        Returns:
            Position-specific rating (1-100)
        """
        position = position.upper()
        
        # Position-specific weightings
        weights = {
            'GK': {'defending': 0.4, 'physical': 0.3, 'pace': 0.2, 'passing': 0.1},
            'CB': {'defending': 0.45, 'physical': 0.25, 'pace': 0.15, 'passing': 0.15},
            'LB': {'defending': 0.35, 'pace': 0.25, 'physical': 0.2, 'passing': 0.2},
            'RB': {'defending': 0.35, 'pace': 0.25, 'physical': 0.2, 'passing': 0.2},
            'LWB': {'pace': 0.3, 'defending': 0.25, 'passing': 0.25, 'physical': 0.2},
            'RWB': {'pace': 0.3, 'defending': 0.25, 'passing': 0.25, 'physical': 0.2},
            'CDM': {'defending': 0.35, 'passing': 0.3, 'physical': 0.25, 'pace': 0.1},
            'CM': {'passing': 0.35, 'defending': 0.25, 'physical': 0.2, 'pace': 0.2},
            'LM': {'passing': 0.3, 'pace': 0.3, 'shooting': 0.2, 'physical': 0.2},
            'RM': {'passing': 0.3, 'pace': 0.3, 'shooting': 0.2, 'physical': 0.2},
            'CAM': {'passing': 0.4, 'shooting': 0.3, 'pace': 0.2, 'physical': 0.1},
            'LW': {'pace': 0.35, 'shooting': 0.3, 'passing': 0.25, 'physical': 0.1},
            'RW': {'pace': 0.35, 'shooting': 0.3, 'passing': 0.25, 'physical': 0.1},
            'ST': {'shooting': 0.4, 'pace': 0.3, 'physical': 0.2, 'passing': 0.1},
            'CF': {'shooting': 0.35, 'passing': 0.3, 'pace': 0.25, 'physical': 0.1},
        }
        
        # Default to overall if position not found
        if position not in weights:
            return self.overall
        
        # Calculate weighted rating
        rating = 0
        weight_dict = weights[position]
        for attr, weight in weight_dict.items():
            rating += getattr(self, attr) * weight
        
        return int(rating)
    
    def is_suited_for_position(self, position: str, threshold: int = 70) -> bool:
        """
        Check if player is suitable for a position.
        
        Args:
            position: Player position
            threshold: Minimum rating required (default: 70)
        
        Returns:
            True if player rating >= threshold for position
        """
        return self.get_position_rating(position) >= threshold


# Example player database (can be extended)
EXAMPLE_PLAYERS = {
    # Arsenal Players
    "Saliba": PlayerStats("William Saliba", pace=75, passing=80, shooting=50, defending=88, physical=82),
    "Gabriel": PlayerStats("Gabriel Magalhaes", pace=72, passing=75, shooting=48, defending=87, physical=85),
    "Rice": PlayerStats("Declan Rice", pace=70, passing=88, shooting=55, defending=85, physical=80),
    "Odegaard": PlayerStats("Martin Odegaard", pace=74, passing=92, shooting=82, defending=65, physical=70),
    "Saka": PlayerStats("Bukayo Saka", pace=86, passing=85, shooting=83, defending=55, physical=72),
    "Jesus": PlayerStats("Gabriel Jesus", pace=85, passing=75, shooting=88, defending=45, physical=75),
    
    # Manchester City
    "Haaland": PlayerStats("Erling Haaland", pace=89, passing=65, shooting=95, defending=35, physical=88),
    "De Bruyne": PlayerStats("Kevin De Bruyne", pace=76, passing=96, shooting=88, defending=62, physical=75),
    "Rodri": PlayerStats("Rodri", pace=62, passing=91, shooting=72, defending=87, physical=82),
    
    # Liverpool
    "Van Dijk": PlayerStats("Virgil van Dijk", pace=77, passing=78, shooting=55, defending=92, physical=88),
    "Salah": PlayerStats("Mohamed Salah", pace=90, passing=84, shooting=91, defending=44, physical=74),
    "Alexander-Arnold": PlayerStats("Trent Alexander-Arnold", pace=76, passing=93, shooting=74, defending=78, physical=72),
    
    # Serie A - Napoli
    "Osimhen": PlayerStats("Victor Osimhen", pace=92, passing=68, shooting=89, defending=38, physical=82),
    "Kvaratskhelia": PlayerStats("Khvicha Kvaratskhelia", pace=88, passing=82, shooting=85, defending=42, physical=70),
    
    # Serie A - Inter Milan
    "Lautaro": PlayerStats("Lautaro Martinez", pace=83, passing=73, shooting=88, defending=48, physical=80),
    "Barella": PlayerStats("Nicolo Barella", pace=78, passing=86, shooting=75, defending=76, physical=77),
    
    # Ligue 1 - PSG
    "Mbappe": PlayerStats("Kylian Mbappe", pace=97, passing=80, shooting=92, defending=36, physical=78),
    "Marquinhos": PlayerStats("Marquinhos", pace=74, passing=77, shooting=52, defending=89, physical=83),
    
    # La Liga - Real Madrid
    "Vinicius": PlayerStats("Vinicius Junior", pace=95, passing=79, shooting=85, defending=32, physical=68),
    "Modric": PlayerStats("Luka Modric", pace=72, passing=94, shooting=76, defending=72, physical=68),
    "Benzema": PlayerStats("Karim Benzema", pace=78, passing=86, shooting=91, defending=40, physical=76),
    
    # La Liga - Barcelona
    "Lewandowski": PlayerStats("Robert Lewandowski", pace=78, passing=80, shooting=93, defending=42, physical=82),
    "Pedri": PlayerStats("Pedri", pace=75, passing=91, shooting=72, defending=66, physical=65),
    "Gavi": PlayerStats("Gavi", pace=77, passing=85, shooting=68, defending=72, physical=70),
    
    # Bundesliga - Bayern Munich
    "Musiala": PlayerStats("Jamal Musiala", pace=82, passing=87, shooting=80, defending=50, physical=65),
    "Kimmich": PlayerStats("Joshua Kimmich", pace=70, passing=92, shooting=74, defending=82, physical=76),
    "Sane": PlayerStats("Leroy Sane", pace=90, passing=83, shooting=86, defending=38, physical=70),
}


def create_player_stats(
    name: str,
    pace: int,
    passing: int,
    shooting: int,
    defending: int,
    physical: int
) -> PlayerStats:
    """
    Factory function to create PlayerStats object.
    
    Args:
        name: Player name
        pace: Pace rating (1-100)
        passing: Passing rating (1-100)
        shooting: Shooting rating (1-100)
        defending: Defending rating (1-100)
        physical: Physical rating (1-100)
    
    Returns:
        PlayerStats object
    """
    return PlayerStats(name, pace, passing, shooting, defending, physical)


def get_player_by_name(name: str) -> PlayerStats:
    """
    Get player stats by player name from example database.
    
    Args:
        name: Player name
    
    Returns:
        PlayerStats object
    
    Raises:
        KeyError: If player not found
    """
    return EXAMPLE_PLAYERS[name]
