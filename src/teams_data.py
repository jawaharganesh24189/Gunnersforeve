"""
Teams data module for football tactics transformer.

This module contains data structures for teams from various leagues,
including team attributes and playing styles.
"""

from typing import Dict, List
from enum import Enum


class League(Enum):
    """Football leagues enumeration"""
    PREMIER_LEAGUE = "Premier League"
    LA_LIGA = "La Liga"
    SERIE_A = "Serie A"
    BUNDESLIGA = "Bundesliga"
    LIGUE_1 = "Ligue 1"


class TeamAttributes:
    """Team attributes and playing style characteristics"""
    
    def __init__(
        self,
        name: str,
        league: League,
        attack_rating: int,
        defense_rating: int,
        possession_style: int,
        pressing_intensity: int,
        preferred_formation: str
    ):
        """
        Initialize team attributes.
        
        Args:
            name: Team name
            league: League the team plays in
            attack_rating: Attacking strength (1-100)
            defense_rating: Defensive strength (1-100)
            possession_style: Possession preference (1-100, higher = more possession-based)
            pressing_intensity: Pressing intensity (1-100, higher = more aggressive)
            preferred_formation: Most commonly used formation
        """
        self.name = name
        self.league = league
        self.attack_rating = attack_rating
        self.defense_rating = defense_rating
        self.possession_style = possession_style
        self.pressing_intensity = pressing_intensity
        self.preferred_formation = preferred_formation
    
    @property
    def overall_rating(self) -> int:
        """Calculate overall team rating"""
        return (self.attack_rating + self.defense_rating) // 2


# Teams database with attributes
TEAMS_DATABASE: Dict[str, TeamAttributes] = {
    # Premier League
    "Arsenal": TeamAttributes("Arsenal", League.PREMIER_LEAGUE, 88, 82, 75, 85, "4-3-3"),
    "Manchester City": TeamAttributes("Manchester City", League.PREMIER_LEAGUE, 92, 85, 88, 90, "4-3-3"),
    "Liverpool": TeamAttributes("Liverpool", League.PREMIER_LEAGUE, 90, 84, 72, 92, "4-3-3"),
    "Manchester United": TeamAttributes("Manchester United", League.PREMIER_LEAGUE, 82, 78, 65, 70, "4-2-3-1"),
    "Chelsea": TeamAttributes("Chelsea", League.PREMIER_LEAGUE, 85, 83, 70, 75, "3-4-3"),
    "Tottenham": TeamAttributes("Tottenham", League.PREMIER_LEAGUE, 84, 76, 68, 78, "4-2-3-1"),
    "Newcastle": TeamAttributes("Newcastle", League.PREMIER_LEAGUE, 78, 82, 62, 75, "4-3-3"),
    "Brighton": TeamAttributes("Brighton", League.PREMIER_LEAGUE, 76, 74, 72, 80, "4-2-3-1"),
    "Aston Villa": TeamAttributes("Aston Villa", League.PREMIER_LEAGUE, 79, 76, 66, 75, "4-2-3-1"),
    "West Ham": TeamAttributes("West Ham", League.PREMIER_LEAGUE, 74, 75, 58, 72, "4-2-3-1"),
    "Fulham": TeamAttributes("Fulham", League.PREMIER_LEAGUE, 73, 72, 64, 70, "4-2-3-1"),
    "Brentford": TeamAttributes("Brentford", League.PREMIER_LEAGUE, 75, 71, 60, 76, "3-5-2"),
    
    # Serie A
    "Juventus": TeamAttributes("Juventus", League.SERIE_A, 84, 88, 68, 72, "3-5-2"),
    "Inter Milan": TeamAttributes("Inter Milan", League.SERIE_A, 86, 87, 70, 75, "3-5-2"),
    "AC Milan": TeamAttributes("AC Milan", League.SERIE_A, 83, 84, 65, 77, "4-2-3-1"),
    "Napoli": TeamAttributes("Napoli", League.SERIE_A, 88, 80, 72, 82, "4-3-3"),
    "Roma": TeamAttributes("Roma", League.SERIE_A, 80, 79, 66, 73, "3-4-2-1"),
    "Lazio": TeamAttributes("Lazio", League.SERIE_A, 81, 77, 64, 74, "4-3-3"),
    "Atalanta": TeamAttributes("Atalanta", League.SERIE_A, 85, 72, 70, 88, "3-4-3"),
    "Fiorentina": TeamAttributes("Fiorentina", League.SERIE_A, 77, 75, 68, 71, "4-3-3"),
    "Bologna": TeamAttributes("Bologna", League.SERIE_A, 76, 74, 65, 73, "4-3-3"),
    "Torino": TeamAttributes("Torino", League.SERIE_A, 74, 76, 62, 71, "3-5-2"),
    "Sassuolo": TeamAttributes("Sassuolo", League.SERIE_A, 75, 70, 67, 72, "4-3-3"),
    "Udinese": TeamAttributes("Udinese", League.SERIE_A, 73, 73, 59, 70, "3-5-2"),
    
    # Ligue 1
    "Paris Saint-Germain": TeamAttributes("Paris Saint-Germain", League.LIGUE_1, 91, 82, 75, 78, "4-3-3"),
    "Marseille": TeamAttributes("Marseille", League.LIGUE_1, 79, 77, 63, 76, "3-4-3"),
    "Monaco": TeamAttributes("Monaco", League.LIGUE_1, 82, 75, 68, 80, "4-4-2"),
    "Lyon": TeamAttributes("Lyon", League.LIGUE_1, 80, 76, 70, 74, "4-3-3"),
    "Lille": TeamAttributes("Lille", League.LIGUE_1, 78, 80, 65, 77, "4-2-3-1"),
    "Rennes": TeamAttributes("Rennes", League.LIGUE_1, 76, 74, 67, 75, "4-3-3"),
    "Nice": TeamAttributes("Nice", League.LIGUE_1, 75, 78, 64, 73, "4-4-2"),
    "Lens": TeamAttributes("Lens", League.LIGUE_1, 77, 76, 66, 79, "3-4-3"),
    "Toulouse": TeamAttributes("Toulouse", League.LIGUE_1, 72, 74, 61, 71, "3-4-3"),
    "Montpellier": TeamAttributes("Montpellier", League.LIGUE_1, 73, 72, 63, 70, "4-2-3-1"),
    "Strasbourg": TeamAttributes("Strasbourg", League.LIGUE_1, 74, 73, 62, 72, "3-5-2"),
    "Nantes": TeamAttributes("Nantes", League.LIGUE_1, 71, 74, 60, 69, "4-4-2"),
    
    # La Liga
    "Real Madrid": TeamAttributes("Real Madrid", League.LA_LIGA, 91, 86, 72, 80, "4-3-3"),
    "Barcelona": TeamAttributes("Barcelona", League.LA_LIGA, 89, 80, 85, 82, "4-3-3"),
    "Atletico Madrid": TeamAttributes("Atletico Madrid", League.LA_LIGA, 82, 89, 62, 88, "3-5-2"),
    "Sevilla": TeamAttributes("Sevilla", League.LA_LIGA, 79, 82, 68, 75, "4-3-3"),
    "Real Sociedad": TeamAttributes("Real Sociedad", League.LA_LIGA, 78, 77, 73, 76, "4-2-3-1"),
    "Real Betis": TeamAttributes("Real Betis", League.LA_LIGA, 77, 74, 71, 74, "4-2-3-1"),
    "Villarreal": TeamAttributes("Villarreal", League.LA_LIGA, 78, 79, 69, 73, "4-4-2"),
    "Athletic Bilbao": TeamAttributes("Athletic Bilbao", League.LA_LIGA, 75, 78, 65, 80, "4-2-3-1"),
    "Valencia": TeamAttributes("Valencia", League.LA_LIGA, 76, 76, 66, 72, "4-4-2"),
    "Celta Vigo": TeamAttributes("Celta Vigo", League.LA_LIGA, 74, 73, 64, 71, "4-1-4-1"),
    "Osasuna": TeamAttributes("Osasuna", League.LA_LIGA, 72, 77, 58, 75, "4-3-3"),
    "Getafe": TeamAttributes("Getafe", League.LA_LIGA, 70, 79, 55, 76, "5-3-2"),
    
    # Bundesliga
    "Bayern Munich": TeamAttributes("Bayern Munich", League.BUNDESLIGA, 93, 84, 78, 87, "4-2-3-1"),
    "Borussia Dortmund": TeamAttributes("Borussia Dortmund", League.BUNDESLIGA, 87, 78, 70, 85, "4-3-3"),
    "RB Leipzig": TeamAttributes("RB Leipzig", League.BUNDESLIGA, 84, 81, 68, 90, "3-4-3"),
    "Bayer Leverkusen": TeamAttributes("Bayer Leverkusen", League.BUNDESLIGA, 82, 77, 71, 82, "4-2-3-1"),
    "Union Berlin": TeamAttributes("Union Berlin", League.BUNDESLIGA, 74, 82, 58, 78, "3-5-2"),
    "Eintracht Frankfurt": TeamAttributes("Eintracht Frankfurt", League.BUNDESLIGA, 79, 76, 66, 81, "3-4-2-1"),
    "Wolfsburg": TeamAttributes("Wolfsburg", League.BUNDESLIGA, 76, 78, 64, 74, "4-2-3-1"),
    "Freiburg": TeamAttributes("Freiburg", League.BUNDESLIGA, 75, 79, 63, 76, "3-4-3"),
    "Borussia Monchengladbach": TeamAttributes("Borussia Monchengladbach", League.BUNDESLIGA, 77, 75, 65, 73, "4-2-3-1"),
    "Mainz": TeamAttributes("Mainz", League.BUNDESLIGA, 73, 74, 62, 72, "3-5-2"),
    "Hoffenheim": TeamAttributes("Hoffenheim", League.BUNDESLIGA, 76, 73, 67, 74, "3-4-3"),
    "Stuttgart": TeamAttributes("Stuttgart", League.BUNDESLIGA, 75, 72, 66, 71, "4-3-3"),
}


def get_team_by_name(team_name: str) -> TeamAttributes:
    """
    Get team attributes by team name.
    
    Args:
        team_name: Name of the team
    
    Returns:
        TeamAttributes object
    
    Raises:
        KeyError: If team not found
    """
    return TEAMS_DATABASE[team_name]


def get_teams_by_league(league: League) -> List[TeamAttributes]:
    """
    Get all teams from a specific league.
    
    Args:
        league: League enum value
    
    Returns:
        List of TeamAttributes for teams in the league
    """
    return [team for team in TEAMS_DATABASE.values() if team.league == league]


def get_all_teams() -> List[TeamAttributes]:
    """
    Get all teams in the database.
    
    Returns:
        List of all TeamAttributes
    """
    return list(TEAMS_DATABASE.values())


def get_team_names() -> List[str]:
    """
    Get list of all team names.
    
    Returns:
        List of team name strings
    """
    return list(TEAMS_DATABASE.keys())
