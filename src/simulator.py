"""
AI-powered football match simulator for Arsenal FC and Premier League.

This module provides intelligent simulation of football matches using
statistical models and machine learning techniques.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from data_schema import MatchData, TeamStats


@dataclass
class TeamProfile:
    """Profile of a team's playing strength and characteristics."""
    name: str
    attack_strength: float  # 0-100
    defense_strength: float  # 0-100
    midfield_strength: float  # 0-100
    form: float  # 0-10 (recent performance)
    home_advantage: float  # 0-20 (boost when playing at home)
    
    @property
    def overall_strength(self) -> float:
        """Calculate overall team strength."""
        return (self.attack_strength * 0.35 + 
                self.defense_strength * 0.30 + 
                self.midfield_strength * 0.35)


# Premier League team profiles (2023-24 season estimates)
PREMIER_LEAGUE_TEAMS = {
    "Arsenal": TeamProfile("Arsenal", 88, 82, 86, 8.5, 12),
    "Manchester City": TeamProfile("Manchester City", 92, 85, 90, 9.0, 10),
    "Liverpool": TeamProfile("Liverpool", 90, 80, 87, 8.0, 11),
    "Manchester United": TeamProfile("Manchester United", 78, 72, 75, 6.5, 11),
    "Chelsea": TeamProfile("Chelsea", 80, 75, 78, 7.0, 10),
    "Tottenham": TeamProfile("Tottenham", 82, 70, 76, 7.5, 10),
    "Newcastle": TeamProfile("Newcastle", 77, 80, 78, 7.8, 12),
    "Brighton": TeamProfile("Brighton", 75, 73, 77, 7.2, 10),
    "Aston Villa": TeamProfile("Aston Villa", 76, 74, 75, 7.0, 11),
    "West Ham": TeamProfile("West Ham", 72, 71, 70, 6.5, 10),
    "Crystal Palace": TeamProfile("Crystal Palace", 68, 72, 69, 6.0, 11),
    "Brentford": TeamProfile("Brentford", 70, 68, 68, 6.5, 12),
    "Fulham": TeamProfile("Fulham", 71, 70, 70, 6.8, 10),
    "Wolves": TeamProfile("Wolves", 67, 73, 68, 6.2, 10),
    "Everton": TeamProfile("Everton", 65, 70, 66, 5.8, 11),
    "Nottingham Forest": TeamProfile("Nottingham Forest", 66, 69, 65, 6.0, 11),
    "Bournemouth": TeamProfile("Bournemouth", 68, 65, 66, 6.0, 10),
    "Luton Town": TeamProfile("Luton Town", 60, 63, 62, 5.5, 12),
    "Burnley": TeamProfile("Burnley", 62, 65, 63, 5.5, 11),
    "Sheffield United": TeamProfile("Sheffield United", 58, 68, 60, 5.0, 11),
}


class FootballMatchSimulator:
    """AI-powered football match simulator."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        date: str,
        competition: str = "Premier League",
        season: str = "2023-24"
    ) -> MatchData:
        """
        Simulate a single match using AI-based algorithms.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            date: Match date (YYYY-MM-DD)
            competition: Competition name
            season: Season identifier
            
        Returns:
            MatchData object with simulated results
        """
        # Get team profiles
        home_profile = PREMIER_LEAGUE_TEAMS.get(
            home_team, 
            TeamProfile(home_team, 70, 70, 70, 6.5, 10)
        )
        away_profile = PREMIER_LEAGUE_TEAMS.get(
            away_team,
            TeamProfile(away_team, 70, 70, 70, 6.5, 10)
        )
        
        # Calculate match outcome using AI model
        home_score, away_score = self._calculate_score(home_profile, away_profile)
        ht_home, ht_away = self._calculate_halftime_score(home_score, away_score)
        
        # Generate match statistics
        home_stats = self._generate_team_stats(
            home_profile, away_profile, home_score, is_home=True
        )
        away_stats = self._generate_team_stats(
            away_profile, home_profile, away_score, is_home=False
        )
        
        # Determine venue
        venue = self._get_venue(home_team)
        attendance = self._simulate_attendance(venue, home_team, away_team)
        
        return MatchData(
            match_id=f"SIM_{season.replace('-', '')}_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}_{date.replace('-', '')}",
            date=date,
            time="15:00",
            competition=competition,
            season=season,
            home_team=home_team,
            away_team=away_team,
            is_arsenal_home=(home_team == "Arsenal"),
            home_score=home_score,
            away_score=away_score,
            halftime_home_score=ht_home,
            halftime_away_score=ht_away,
            venue=venue,
            attendance=attendance,
            home_stats=home_stats,
            away_stats=away_stats,
        )
    
    def _calculate_score(
        self, 
        home_profile: TeamProfile, 
        away_profile: TeamProfile
    ) -> Tuple[int, int]:
        """
        Calculate match score using Poisson distribution based on team strengths.
        
        This simulates realistic football scores where goals are relatively rare events.
        """
        # Calculate expected goals (xG) for each team
        home_strength = home_profile.attack_strength + home_profile.home_advantage
        away_strength = away_profile.attack_strength
        
        home_defense_factor = away_profile.defense_strength / 100
        away_defense_factor = home_profile.defense_strength / 100
        
        # Base goal expectation (league average ~1.4 goals per team)
        base_xg = 1.4
        
        # Calculate xG using strength differentials
        home_xg = base_xg * (home_strength / 80) * (1 - home_defense_factor * 0.5)
        away_xg = base_xg * (away_strength / 80) * (1 - away_defense_factor * 0.5)
        
        # Add form factor
        home_xg *= (1 + (home_profile.form - 6.5) * 0.05)
        away_xg *= (1 + (away_profile.form - 6.5) * 0.05)
        
        # Sample from Poisson distribution
        home_score = np.random.poisson(max(0.3, home_xg))
        away_score = np.random.poisson(max(0.3, away_xg))
        
        return int(home_score), int(away_score)
    
    def _calculate_halftime_score(
        self, 
        full_home: int, 
        full_away: int
    ) -> Tuple[int, int]:
        """Calculate realistic halftime score."""
        # On average, 45% of goals are scored in first half
        ht_home = int(full_home * random.uniform(0.3, 0.6))
        ht_away = int(full_away * random.uniform(0.3, 0.6))
        
        return ht_home, ht_away
    
    def _generate_team_stats(
        self,
        team_profile: TeamProfile,
        opponent_profile: TeamProfile,
        goals: int,
        is_home: bool
    ) -> TeamStats:
        """Generate realistic match statistics for a team."""
        # Base statistics correlated with team strength and goals
        strength = team_profile.overall_strength
        opp_strength = opponent_profile.overall_strength
        
        # Possession (correlated with midfield strength)
        base_possession = 50 + (team_profile.midfield_strength - opponent_profile.midfield_strength) * 0.3
        possession = max(30, min(70, base_possession + random.uniform(-5, 5)))
        
        # Shots (correlated with attack strength and goals)
        base_shots = 10 + (team_profile.attack_strength / 10) + (goals * 2)
        shots = int(max(5, base_shots + random.uniform(-3, 3)))
        
        # Shots on target (35-50% of shots)
        shots_on_target = int(shots * random.uniform(0.35, 0.50))
        shots_on_target = max(goals, shots_on_target)  # At least as many as goals
        
        # Expected goals (xG) - realistic based on shots
        xg = round(shots_on_target * random.uniform(0.10, 0.18), 2)
        
        # Corners (correlated with possession and attack)
        corners = int(max(0, 4 + (possession - 50) / 10 + random.uniform(-2, 2)))
        
        # Fouls (inversely correlated with possession)
        fouls = int(max(5, 12 - (possession - 50) / 10 + random.uniform(-2, 2)))
        
        # Cards
        yellow_cards = int(max(0, fouls / 6 + random.uniform(-0.5, 1)))
        red_cards = 1 if random.random() < 0.05 else 0  # 5% chance of red card
        
        return TeamStats(
            team_name=team_profile.name,
            goals=goals,
            shots=shots,
            shots_on_target=shots_on_target,
            possession=round(possession, 1),
            corners=corners,
            fouls=fouls,
            yellow_cards=yellow_cards,
            red_cards=red_cards,
            xg=xg
        )
    
    def _get_venue(self, home_team: str) -> str:
        """Get the home venue for a team."""
        venues = {
            "Arsenal": "Emirates Stadium",
            "Manchester City": "Etihad Stadium",
            "Liverpool": "Anfield",
            "Manchester United": "Old Trafford",
            "Chelsea": "Stamford Bridge",
            "Tottenham": "Tottenham Hotspur Stadium",
            "Newcastle": "St. James' Park",
            "Brighton": "Amex Stadium",
            "Aston Villa": "Villa Park",
            "West Ham": "London Stadium",
        }
        return venues.get(home_team, f"{home_team} Stadium")
    
    def _simulate_attendance(
        self, 
        venue: str, 
        home_team: str, 
        away_team: str
    ) -> int:
        """Simulate realistic attendance figures."""
        # Stadium capacities (approximate)
        capacities = {
            "Emirates Stadium": 60704,
            "Etihad Stadium": 53400,
            "Anfield": 53394,
            "Old Trafford": 74140,
            "Stamford Bridge": 40341,
            "Tottenham Hotspur Stadium": 62850,
        }
        
        capacity = capacities.get(venue, 30000)
        
        # Big matches draw more crowds (90-100% capacity)
        # Regular matches draw 85-95% capacity
        big_teams = ["Arsenal", "Manchester City", "Liverpool", "Manchester United", "Chelsea"]
        
        if home_team in big_teams and away_team in big_teams:
            fill_rate = random.uniform(0.95, 0.99)
        else:
            fill_rate = random.uniform(0.85, 0.95)
        
        return int(capacity * fill_rate)
    
    def simulate_season(
        self,
        team: str = "Arsenal",
        season: str = "2023-24",
        num_matches: int = 38
    ) -> List[MatchData]:
        """
        Simulate a full season for a team.
        
        Args:
            team: Team to simulate (default: Arsenal)
            season: Season identifier
            num_matches: Number of matches to simulate
            
        Returns:
            List of simulated matches
        """
        matches = []
        opponents = [t for t in PREMIER_LEAGUE_TEAMS.keys() if t != team]
        
        # Sample opponents (with replacement for full season)
        season_opponents = random.choices(opponents, k=num_matches)
        
        # Generate dates
        start_date = datetime(2023, 8, 12)
        
        for i, opponent in enumerate(season_opponents):
            # Alternate home/away (roughly)
            is_home = (i % 2 == 0)
            home_team = team if is_home else opponent
            away_team = opponent if is_home else team
            
            # Match date (roughly weekly)
            match_date = start_date + timedelta(days=i * 7)
            
            match = self.simulate_match(
                home_team=home_team,
                away_team=away_team,
                date=match_date.strftime("%Y-%m-%d"),
                season=season
            )
            matches.append(match)
        
        return matches
    
    def simulate_league_round(
        self,
        date: str,
        season: str = "2023-24"
    ) -> List[MatchData]:
        """
        Simulate a full round of Premier League matches.
        
        Args:
            date: Match date for the round
            season: Season identifier
            
        Returns:
            List of simulated matches (typically 10 matches)
        """
        teams = list(PREMIER_LEAGUE_TEAMS.keys())
        random.shuffle(teams)
        
        matches = []
        # Create 10 matches (20 teams)
        for i in range(0, len(teams) - 1, 2):
            match = self.simulate_match(
                home_team=teams[i],
                away_team=teams[i + 1],
                date=date,
                season=season
            )
            matches.append(match)
        
        return matches


def create_simulated_dataset(
    num_matches: int = 50,
    team: str = "Arsenal",
    season: str = "2023-24",
    seed: Optional[int] = 42
) -> 'Dataset':
    """
    Create a dataset of simulated matches.
    
    Args:
        num_matches: Number of matches to simulate
        team: Team to focus on
        season: Season identifier
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with simulated matches
    """
    from data_schema import Dataset
    
    simulator = FootballMatchSimulator(seed=seed)
    matches = simulator.simulate_season(team=team, season=season, num_matches=num_matches)
    
    return Dataset(
        dataset_name=f"Simulated {team} Matches {season}",
        description=f"AI-simulated {team} match data for {season} season using statistical models",
        source="AI Simulation Engine v1.0",
        last_updated=datetime.now().isoformat(),
        matches=matches
    )


if __name__ == "__main__":
    # Example usage
    print("Creating AI-simulated Arsenal match dataset...")
    
    # Create simulator
    simulator = FootballMatchSimulator(seed=42)
    
    # Simulate a single match
    match = simulator.simulate_match(
        home_team="Arsenal",
        away_team="Manchester City",
        date="2023-10-08"
    )
    
    print(f"\nSimulated Match:")
    print(f"{match.home_team} {match.home_score} - {match.away_score} {match.away_team}")
    print(f"Possession: {match.home_stats.possession}% - {match.away_stats.possession}%")
    print(f"Shots: {match.home_stats.shots} - {match.away_stats.shots}")
    print(f"xG: {match.home_stats.xg} - {match.away_stats.xg}")
    
    # Simulate a season
    print(f"\nSimulating 10 Arsenal matches...")
    dataset = create_simulated_dataset(num_matches=10, team="Arsenal", season="2023-24")
    
    print(f"Created dataset: {dataset.dataset_name}")
    print(f"Total matches: {len(dataset.matches)}")
    
    # Calculate results
    wins = sum(1 for m in dataset.matches if (
        (m.is_arsenal_home and m.home_score > m.away_score) or
        (not m.is_arsenal_home and m.away_score > m.home_score)
    ))
    draws = sum(1 for m in dataset.matches if m.home_score == m.away_score)
    losses = len(dataset.matches) - wins - draws
    
    print(f"Record: {wins}W {draws}D {losses}L")
