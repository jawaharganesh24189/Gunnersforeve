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
    "Ramsdale": PlayerStats("Aaron Ramsdale", pace=55, passing=60, shooting=40, defending=85, physical=78),
    "White": PlayerStats("Ben White", pace=78, passing=82, shooting=52, defending=84, physical=76),
    "Partey": PlayerStats("Thomas Partey", pace=74, passing=85, shooting=68, defending=83, physical=82),
    
    # Manchester City
    "Haaland": PlayerStats("Erling Haaland", pace=89, passing=65, shooting=95, defending=35, physical=88),
    "De Bruyne": PlayerStats("Kevin De Bruyne", pace=76, passing=96, shooting=88, defending=62, physical=75),
    "Rodri": PlayerStats("Rodri", pace=62, passing=91, shooting=72, defending=87, physical=82),
    "Ederson": PlayerStats("Ederson", pace=60, passing=85, shooting=45, defending=88, physical=80),
    "Grealish": PlayerStats("Jack Grealish", pace=83, passing=87, shooting=76, defending=48, physical=68),
    "Bernardo Silva": PlayerStats("Bernardo Silva", pace=80, passing=91, shooting=80, defending=65, physical=70),
    
    # Liverpool
    "Van Dijk": PlayerStats("Virgil van Dijk", pace=77, passing=78, shooting=55, defending=92, physical=88),
    "Salah": PlayerStats("Mohamed Salah", pace=90, passing=84, shooting=91, defending=44, physical=74),
    "Alexander-Arnold": PlayerStats("Trent Alexander-Arnold", pace=76, passing=93, shooting=74, defending=78, physical=72),
    "Alisson": PlayerStats("Alisson Becker", pace=58, passing=75, shooting=42, defending=92, physical=85),
    "Diaz": PlayerStats("Luis Diaz", pace=91, passing=80, shooting=82, defending=40, physical=72),
    "Mac Allister": PlayerStats("Alexis Mac Allister", pace=75, passing=88, shooting=77, defending=75, physical=74),
    
    # Chelsea
    "James": PlayerStats("Reece James", pace=82, passing=86, shooting=78, defending=82, physical=80),
    "Sterling": PlayerStats("Raheem Sterling", pace=88, passing=82, shooting=84, defending=42, physical=70),
    "Enzo": PlayerStats("Enzo Fernandez", pace=74, passing=89, shooting=74, defending=76, physical=73),
    
    # Manchester United
    "Rashford": PlayerStats("Marcus Rashford", pace=90, passing=79, shooting=87, defending=40, physical=75),
    "Bruno": PlayerStats("Bruno Fernandes", pace=76, passing=90, shooting=85, defending=60, physical=72),
    "Casemiro": PlayerStats("Casemiro", pace=68, passing=84, shooting=70, defending=88, physical=84),
    
    # Tottenham
    "Son": PlayerStats("Son Heung-min", pace=87, passing=82, shooting=89, defending=44, physical=73),
    "Kane": PlayerStats("Harry Kane", pace=70, passing=87, shooting=93, defending=48, physical=80),
    "Kulusevski": PlayerStats("Dejan Kulusevski", pace=82, passing=84, shooting=79, defending=52, physical=76),
    
    # Serie A - Napoli
    "Osimhen": PlayerStats("Victor Osimhen", pace=92, passing=68, shooting=89, defending=38, physical=82),
    "Kvaratskhelia": PlayerStats("Khvicha Kvaratskhelia", pace=88, passing=82, shooting=85, defending=42, physical=70),
    "Kim": PlayerStats("Kim Min-jae", pace=79, passing=76, shooting=48, defending=89, physical=85),
    
    # Serie A - Inter Milan
    "Lautaro": PlayerStats("Lautaro Martinez", pace=83, passing=73, shooting=88, defending=48, physical=80),
    "Barella": PlayerStats("Nicolo Barella", pace=78, passing=86, shooting=75, defending=76, physical=77),
    "Bastoni": PlayerStats("Alessandro Bastoni", pace=75, passing=84, shooting=50, defending=87, physical=79),
    
    # Serie A - AC Milan
    "Leao": PlayerStats("Rafael Leao", pace=93, passing=78, shooting=82, defending=36, physical=75),
    "Tonali": PlayerStats("Sandro Tonali", pace=73, passing=83, shooting=70, defending=81, physical=78),
    "Maignan": PlayerStats("Mike Maignan", pace=62, passing=68, shooting=40, defending=90, physical=82),
    
    # Serie A - Juventus
    "Vlahovic": PlayerStats("Dusan Vlahovic", pace=80, passing=70, shooting=90, defending=42, physical=82),
    "Chiesa": PlayerStats("Federico Chiesa", pace=89, passing=81, shooting=84, defending=45, physical=74),
    "Bremer": PlayerStats("Bremer", pace=76, passing=72, shooting=46, defending=90, physical=86),
    
    # Ligue 1 - PSG
    "Mbappe": PlayerStats("Kylian Mbappe", pace=97, passing=80, shooting=92, defending=36, physical=78),
    "Marquinhos": PlayerStats("Marquinhos", pace=74, passing=77, shooting=52, defending=89, physical=83),
    "Hakimi": PlayerStats("Achraf Hakimi", pace=93, passing=81, shooting=74, defending=76, physical=78),
    "Verratti": PlayerStats("Marco Verratti", pace=72, passing=92, shooting=68, defending=74, physical=70),
    
    # Ligue 1 - Marseille
    "Alexis": PlayerStats("Alexis Sanchez", pace=82, passing=84, shooting=86, defending=42, physical=76),
    "Clauss": PlayerStats("Jonathan Clauss", pace=84, passing=80, shooting=70, defending=75, physical=74),
    
    # La Liga - Real Madrid
    "Vinicius": PlayerStats("Vinicius Junior", pace=95, passing=79, shooting=85, defending=32, physical=68),
    "Modric": PlayerStats("Luka Modric", pace=72, passing=94, shooting=76, defending=72, physical=68),
    "Benzema": PlayerStats("Karim Benzema", pace=78, passing=86, shooting=91, defending=40, physical=76),
    "Courtois": PlayerStats("Thibaut Courtois", pace=56, passing=70, shooting=38, defending=93, physical=88),
    "Bellingham": PlayerStats("Jude Bellingham", pace=77, passing=86, shooting=81, defending=73, physical=78),
    "Rudiger": PlayerStats("Antonio Rudiger", pace=80, passing=75, shooting=48, defending=88, physical=85),
    
    # La Liga - Barcelona
    "Lewandowski": PlayerStats("Robert Lewandowski", pace=78, passing=80, shooting=93, defending=42, physical=82),
    "Pedri": PlayerStats("Pedri", pace=75, passing=91, shooting=72, defending=66, physical=65),
    "Gavi": PlayerStats("Gavi", pace=77, passing=85, shooting=68, defending=72, physical=70),
    "Ter Stegen": PlayerStats("Marc-Andre ter Stegen", pace=58, passing=82, shooting=40, defending=91, physical=82),
    "De Jong": PlayerStats("Frenkie de Jong", pace=78, passing=90, shooting=72, defending=77, physical=75),
    "Araujo": PlayerStats("Ronald Araujo", pace=78, passing=73, shooting=50, defending=89, physical=84),
    
    # La Liga - Atletico Madrid
    "Griezmann": PlayerStats("Antoine Griezmann", pace=80, passing=87, shooting=87, defending=56, physical=72),
    "Oblak": PlayerStats("Jan Oblak", pace=54, passing=65, shooting=36, defending=94, physical=86),
    "Gimenez": PlayerStats("Jose Maria Gimenez", pace=76, passing=70, shooting=46, defending=90, physical=87),
    
    # Bundesliga - Bayern Munich
    "Musiala": PlayerStats("Jamal Musiala", pace=82, passing=87, shooting=80, defending=50, physical=65),
    "Kimmich": PlayerStats("Joshua Kimmich", pace=70, passing=92, shooting=74, defending=82, physical=76),
    "Sane": PlayerStats("Leroy Sane", pace=90, passing=83, shooting=86, defending=38, physical=70),
    "Neuer": PlayerStats("Manuel Neuer", pace=57, passing=84, shooting=38, defending=92, physical=85),
    "Davies": PlayerStats("Alphonso Davies", pace=96, passing=76, shooting=64, defending=78, physical=77),
    "Coman": PlayerStats("Kingsley Coman", pace=91, passing=80, shooting=80, defending=40, physical=70),
    
    # Bundesliga - Borussia Dortmund
    "Bellingham_BVB": PlayerStats("Jude Bellingham", pace=77, passing=86, shooting=81, defending=73, physical=78),
    "Reus": PlayerStats("Marco Reus", pace=79, passing=88, shooting=84, defending=54, physical=68),
    "Hummels": PlayerStats("Mats Hummels", pace=68, passing=80, shooting=52, defending=88, physical=80),
    "Brandt": PlayerStats("Julian Brandt", pace=80, passing=86, shooting=78, defending=58, physical=68),
    
    # Bundesliga - RB Leipzig
    "Nkunku": PlayerStats("Christopher Nkunku", pace=85, passing=84, shooting=88, defending=52, physical=72),
    "Szoboszlai": PlayerStats("Dominik Szoboszlai", pace=80, passing=87, shooting=82, defending=65, physical=74),
    
    # Bundesliga - Bayer Leverkusen
    "Wirtz": PlayerStats("Florian Wirtz", pace=79, passing=88, shooting=82, defending=56, physical=66),
    "Frimpong": PlayerStats("Jeremie Frimpong", pace=92, passing=78, shooting=72, defending=74, physical=75),
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
