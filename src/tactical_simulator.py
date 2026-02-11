"""
Advanced Tactical Football Match Simulator with Event-Level Dynamics.

This module provides sophisticated match simulation with:
- Tactical formations and playing styles
- Event-by-event simulation (minute-by-minute)
- Match momentum and psychology
- Substitutions and their impact
- Detailed player interactions
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from data_schema import MatchData, TeamStats


class Formation(Enum):
    """Football formations."""
    F_4_3_3 = "4-3-3"
    F_4_4_2 = "4-4-2"
    F_3_5_2 = "3-5-2"
    F_4_2_3_1 = "4-2-3-1"
    F_3_4_3 = "3-4-3"
    F_5_3_2 = "5-3-2"


class PlayingStyle(Enum):
    """Team playing styles."""
    POSSESSION = "possession"
    COUNTER_ATTACK = "counter_attack"
    HIGH_PRESS = "high_press"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    DIRECT = "direct"


class EventType(Enum):
    """Match event types."""
    KICK_OFF = "kick_off"
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    SAVE = "save"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    TACKLE = "tackle"
    FOUL = "foul"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"


@dataclass
class MatchEvent:
    """Individual match event."""
    minute: int
    event_type: EventType
    team: str
    player: Optional[str] = None
    description: str = ""
    x_position: Optional[float] = None  # 0-100 (0=own goal, 100=opponent goal)
    y_position: Optional[float] = None  # 0-100 (field width)
    outcome: Optional[str] = None  # success, failure, etc.


@dataclass
class TacticalSetup:
    """Team tactical setup."""
    formation: Formation
    style: PlayingStyle
    line_height: float = 50.0  # 0-100, defensive line position
    pressing_intensity: float = 50.0  # 0-100
    width: float = 50.0  # 0-100, how wide the team plays
    tempo: float = 50.0  # 0-100, speed of play


@dataclass
class MatchState:
    """Current state of the match."""
    minute: int = 0
    home_score: int = 0
    away_score: int = 0
    home_possession: float = 50.0
    away_possession: float = 50.0
    momentum: float = 0.0  # -100 to 100 (negative = away, positive = home)
    home_energy: float = 100.0  # Team energy level
    away_energy: float = 100.0
    home_morale: float = 50.0  # Team morale
    away_morale: float = 50.0
    events: List[MatchEvent] = field(default_factory=list)
    
    # Statistics
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_fouls: int = 0
    away_fouls: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_passes: int = 0
    away_passes: int = 0
    home_successful_passes: int = 0
    away_successful_passes: int = 0
    home_tackles: int = 0
    away_tackles: int = 0


@dataclass
class AdvancedTeamProfile:
    """Enhanced team profile with tactical attributes."""
    name: str
    
    # Core attributes
    attack_strength: float  # 0-100
    defense_strength: float  # 0-100
    midfield_strength: float  # 0-100
    
    # Tactical attributes
    pressing_ability: float = 70.0  # 0-100
    passing_quality: float = 70.0  # 0-100
    pace: float = 70.0  # 0-100
    physicality: float = 70.0  # 0-100
    creativity: float = 70.0  # 0-100
    discipline: float = 70.0  # 0-100
    
    # Situational
    form: float = 7.0  # 0-10
    home_advantage: float = 12.0  # 0-20
    
    # Preferred tactics
    preferred_formation: Formation = Formation.F_4_3_3
    preferred_style: PlayingStyle = PlayingStyle.BALANCED
    
    @property
    def overall_strength(self) -> float:
        """Calculate overall team strength."""
        return (self.attack_strength * 0.35 + 
                self.defense_strength * 0.30 + 
                self.midfield_strength * 0.35)


# Enhanced Premier League team profiles with tactical attributes
ADVANCED_PL_TEAMS = {
    "Arsenal": AdvancedTeamProfile(
        name="Arsenal",
        attack_strength=88, defense_strength=82, midfield_strength=86,
        pressing_ability=85, passing_quality=88, pace=84,
        physicality=78, creativity=87, discipline=82,
        form=8.5, home_advantage=12,
        preferred_formation=Formation.F_4_3_3,
        preferred_style=PlayingStyle.POSSESSION
    ),
    "Manchester City": AdvancedTeamProfile(
        name="Manchester City",
        attack_strength=92, defense_strength=85, midfield_strength=90,
        pressing_ability=90, passing_quality=92, pace=82,
        physicality=80, creativity=90, discipline=85,
        form=9.0, home_advantage=10,
        preferred_formation=Formation.F_4_3_3,
        preferred_style=PlayingStyle.POSSESSION
    ),
    "Liverpool": AdvancedTeamProfile(
        name="Liverpool",
        attack_strength=90, defense_strength=80, midfield_strength=87,
        pressing_ability=92, passing_quality=85, pace=88,
        physicality=85, creativity=84, discipline=80,
        form=8.0, home_advantage=11,
        preferred_formation=Formation.F_4_3_3,
        preferred_style=PlayingStyle.HIGH_PRESS
    ),
    "Manchester United": AdvancedTeamProfile(
        name="Manchester United",
        attack_strength=78, defense_strength=72, midfield_strength=75,
        pressing_ability=70, passing_quality=75, pace=80,
        physicality=82, creativity=76, discipline=70,
        form=6.5, home_advantage=11,
        preferred_formation=Formation.F_4_2_3_1,
        preferred_style=PlayingStyle.COUNTER_ATTACK
    ),
    "Chelsea": AdvancedTeamProfile(
        name="Chelsea",
        attack_strength=80, defense_strength=75, midfield_strength=78,
        pressing_ability=75, passing_quality=78, pace=79,
        physicality=80, creativity=77, discipline=76,
        form=7.0, home_advantage=10,
        preferred_formation=Formation.F_4_2_3_1,
        preferred_style=PlayingStyle.BALANCED
    ),
    "Tottenham": AdvancedTeamProfile(
        name="Tottenham",
        attack_strength=82, defense_strength=70, midfield_strength=76,
        pressing_ability=72, passing_quality=76, pace=85,
        physicality=78, creativity=80, discipline=68,
        form=7.5, home_advantage=10,
        preferred_formation=Formation.F_4_2_3_1,
        preferred_style=PlayingStyle.COUNTER_ATTACK
    ),
    "Newcastle": AdvancedTeamProfile(
        name="Newcastle",
        attack_strength=77, defense_strength=80, midfield_strength=78,
        pressing_ability=82, passing_quality=74, pace=80,
        physicality=85, creativity=72, discipline=80,
        form=7.8, home_advantage=12,
        preferred_formation=Formation.F_4_3_3,
        preferred_style=PlayingStyle.HIGH_PRESS
    ),
}


class TacticalMatchSimulator:
    """Advanced tactical match simulator with event-level dynamics."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize tactical simulator."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_state: Optional[MatchState] = None
    
    def simulate_tactical_match(
        self,
        home_team: str,
        away_team: str,
        date: str,
        competition: str = "Premier League",
        season: str = "2023-24",
        detailed_events: bool = True
    ) -> Tuple[MatchData, MatchState]:
        """
        Simulate a match with full tactical and event dynamics.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            date: Match date
            competition: Competition name
            season: Season identifier
            detailed_events: Whether to generate detailed event log
            
        Returns:
            Tuple of (MatchData, MatchState with event log)
        """
        # Get team profiles
        home_profile = ADVANCED_PL_TEAMS.get(
            home_team,
            AdvancedTeamProfile(home_team, 70, 70, 70)
        )
        away_profile = ADVANCED_PL_TEAMS.get(
            away_team,
            AdvancedTeamProfile(away_team, 70, 70, 70)
        )
        
        # Set up tactics
        home_tactics = TacticalSetup(
            formation=home_profile.preferred_formation,
            style=home_profile.preferred_style,
            line_height=self._calculate_line_height(home_profile, away_profile),
            pressing_intensity=home_profile.pressing_ability,
            width=50 + (home_profile.pace - 70) * 0.3,
            tempo=50 + (home_profile.form - 6.5) * 5
        )
        
        away_tactics = TacticalSetup(
            formation=away_profile.preferred_formation,
            style=away_profile.preferred_style,
            line_height=self._calculate_line_height(away_profile, home_profile),
            pressing_intensity=away_profile.pressing_ability,
            width=50 + (away_profile.pace - 70) * 0.3,
            tempo=50 + (away_profile.form - 6.5) * 5
        )
        
        # Initialize match state
        self.current_state = MatchState()
        
        # Simulate match minute by minute
        if detailed_events:
            self._simulate_match_events(
                home_profile, away_profile,
                home_tactics, away_tactics
            )
        else:
            # Quick simulation without detailed events
            self._simulate_match_quick(home_profile, away_profile)
        
        # Create match data from final state
        match_data = self._create_match_data(
            home_team, away_team, date, competition, season,
            home_profile, away_profile
        )
        
        return match_data, self.current_state
    
    def _simulate_match_events(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile,
        home_tactics: TacticalSetup,
        away_tactics: TacticalSetup
    ):
        """Simulate match minute by minute with events."""
        
        # Kick-off
        self.current_state.events.append(MatchEvent(
            minute=0, event_type=EventType.KICK_OFF,
            team=home_profile.name, description="Match starts"
        ))
        
        # First half (45 minutes + injury time)
        first_half_minutes = 45 + random.randint(0, 3)
        for minute in range(1, first_half_minutes + 1):
            self.current_state.minute = minute
            self._simulate_minute(
                home_profile, away_profile,
                home_tactics, away_tactics,
                is_first_half=True
            )
        
        # Half-time
        self.current_state.events.append(MatchEvent(
            minute=45, event_type=EventType.HALF_TIME,
            team="", description="Half-time"
        ))
        
        # Adjust tactics at half-time based on score
        self._adjust_half_time_tactics(
            home_profile, away_profile,
            home_tactics, away_tactics
        )
        
        # Second half (45 minutes + injury time)
        second_half_minutes = 45 + random.randint(2, 5)
        for minute in range(46, 46 + second_half_minutes):
            self.current_state.minute = minute
            self._simulate_minute(
                home_profile, away_profile,
                home_tactics, away_tactics,
                is_first_half=False
            )
        
        # Full-time
        self.current_state.events.append(MatchEvent(
            minute=90, event_type=EventType.FULL_TIME,
            team="", description="Full-time"
        ))
    
    def _simulate_minute(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile,
        home_tactics: TacticalSetup,
        away_tactics: TacticalSetup,
        is_first_half: bool
    ):
        """Simulate a single minute of play."""
        
        # Decay energy and adjust based on fitness
        energy_decay = 0.15 + (100 - home_profile.physicality) * 0.01
        self.current_state.home_energy = max(20, self.current_state.home_energy - energy_decay)
        self.current_state.away_energy = max(20, self.current_state.away_energy - energy_decay)
        
        # Calculate possession for this minute based on tactics and momentum
        minute_possession = self._calculate_minute_possession(
            home_profile, away_profile,
            home_tactics, away_tactics
        )
        
        # Update possession averages
        self.current_state.home_possession = (
            self.current_state.home_possession * 0.95 + minute_possession * 0.05
        )
        self.current_state.away_possession = 100 - self.current_state.home_possession
        
        # Determine which team has the ball
        has_ball_home = random.random() < (minute_possession / 100)
        attacking_team = home_profile if has_ball_home else away_profile
        defending_team = away_profile if has_ball_home else home_profile
        attacking_tactics = home_tactics if has_ball_home else away_tactics
        defending_tactics = away_tactics if has_ball_home else home_tactics
        
        # Chance of creating an attack (depends on tempo and quality)
        attack_chance = (attacking_team.attack_strength + attacking_team.midfield_strength) / 200
        attack_chance *= (attacking_tactics.tempo / 50)
        
        if random.random() < attack_chance * 0.3:  # ~30% of possessions lead to attacks
            self._simulate_attack(
                attacking_team, defending_team,
                attacking_tactics, defending_tactics,
                has_ball_home
            )
    
    def _simulate_attack(
        self,
        attacking_team: AdvancedTeamProfile,
        defending_team: AdvancedTeamProfile,
        attacking_tactics: TacticalSetup,
        defending_tactics: TacticalSetup,
        is_home_attacking: bool
    ):
        """Simulate an attacking phase."""
        
        # Calculate attack quality
        attack_quality = (
            attacking_team.attack_strength * 0.4 +
            attacking_team.creativity * 0.3 +
            attacking_team.passing_quality * 0.3
        ) / 100
        
        # Calculate defense quality
        defense_quality = (
            defending_team.defense_strength * 0.5 +
            defending_team.physicality * 0.3 +
            defending_tactics.pressing_intensity * 0.2
        ) / 100
        
        # Energy affects both
        if is_home_attacking:
            attack_quality *= (self.current_state.home_energy / 100)
            defense_quality *= (self.current_state.away_energy / 100)
        else:
            attack_quality *= (self.current_state.away_energy / 100)
            defense_quality *= (self.current_state.home_energy / 100)
        
        # Determine attack outcome
        outcome_roll = random.random()
        
        if outcome_roll < defense_quality * 0.3:
            # Tackle/Interception
            if is_home_attacking:
                self.current_state.away_tackles += 1
            else:
                self.current_state.home_tackles += 1
                
        elif outcome_roll < defense_quality * 0.5:
            # Foul
            self._simulate_foul(attacking_team, defending_team, is_home_attacking)
            
        elif outcome_roll < 0.7:
            # Attack breaks down, possible corner
            if random.random() < 0.3:
                self._simulate_corner(attacking_team, is_home_attacking)
                
        else:
            # Shot opportunity
            self._simulate_shot(
                attacking_team, defending_team,
                attacking_tactics, is_home_attacking,
                attack_quality
            )
    
    def _simulate_shot(
        self,
        attacking_team: AdvancedTeamProfile,
        defending_team: AdvancedTeamProfile,
        attacking_tactics: TacticalSetup,
        is_home_attacking: bool,
        attack_quality: float
    ):
        """Simulate a shot attempt."""
        
        if is_home_attacking:
            self.current_state.home_shots += 1
        else:
            self.current_state.away_shots += 1
        
        # Shot quality based on position and pressure
        shot_quality = attack_quality * random.uniform(0.6, 1.0)
        
        # On target probability
        on_target_prob = 0.3 + shot_quality * 0.3
        
        if random.random() < on_target_prob:
            # Shot on target
            if is_home_attacking:
                self.current_state.home_shots_on_target += 1
            else:
                self.current_state.away_shots_on_target += 1
            
            # Goal probability (typical ~15-20% of shots on target)
            goal_prob = 0.12 + shot_quality * 0.15
            
            # Adjust for goalkeeper (inverse of defending team's defense)
            goalkeeper_save_prob = defending_team.defense_strength / 100 * 0.7
            goal_prob *= (1 - goalkeeper_save_prob * 0.5)
            
            if random.random() < goal_prob:
                # GOAL!
                self._score_goal(attacking_team, is_home_attacking)
            else:
                # Save
                self.current_state.events.append(MatchEvent(
                    minute=self.current_state.minute,
                    event_type=EventType.SAVE,
                    team=defending_team.name,
                    description=f"Save by {defending_team.name} goalkeeper"
                ))
        else:
            # Shot off target
            self.current_state.events.append(MatchEvent(
                minute=self.current_state.minute,
                event_type=EventType.SHOT,
                team=attacking_team.name,
                description=f"Shot off target by {attacking_team.name}",
                outcome="miss"
            ))
    
    def _score_goal(self, team: AdvancedTeamProfile, is_home: bool):
        """Register a goal."""
        if is_home:
            self.current_state.home_score += 1
        else:
            self.current_state.away_score += 1
        
        self.current_state.events.append(MatchEvent(
            minute=self.current_state.minute,
            event_type=EventType.GOAL,
            team=team.name,
            description=f"⚽ GOAL! {team.name} scores!",
            outcome="success"
        ))
        
        # Adjust momentum and morale
        if is_home:
            self.current_state.momentum += 20
            self.current_state.home_morale = min(100, self.current_state.home_morale + 15)
            self.current_state.away_morale = max(20, self.current_state.away_morale - 10)
        else:
            self.current_state.momentum -= 20
            self.current_state.away_morale = min(100, self.current_state.away_morale + 15)
            self.current_state.home_morale = max(20, self.current_state.home_morale - 10)
    
    def _simulate_corner(self, team: AdvancedTeamProfile, is_home: bool):
        """Simulate a corner kick."""
        if is_home:
            self.current_state.home_corners += 1
        else:
            self.current_state.away_corners += 1
        
        self.current_state.events.append(MatchEvent(
            minute=self.current_state.minute,
            event_type=EventType.CORNER,
            team=team.name,
            description=f"Corner for {team.name}"
        ))
    
    def _simulate_foul(
        self,
        attacking_team: AdvancedTeamProfile,
        defending_team: AdvancedTeamProfile,
        is_home_attacking: bool
    ):
        """Simulate a foul."""
        if is_home_attacking:
            self.current_state.away_fouls += 1
        else:
            self.current_state.home_fouls += 1
        
        # Card probability (depends on discipline and existing cards)
        card_prob = (100 - defending_team.discipline) / 1000
        
        if random.random() < card_prob:
            # Yellow card
            if is_home_attacking:
                self.current_state.away_yellow_cards += 1
            else:
                self.current_state.home_yellow_cards += 1
            
            self.current_state.events.append(MatchEvent(
                minute=self.current_state.minute,
                event_type=EventType.YELLOW_CARD,
                team=defending_team.name,
                description=f"🟨 Yellow card for {defending_team.name}"
            ))
            
            # Red card (rare, ~5% of yellow card situations if already on yellow)
            if random.random() < 0.05:
                if is_home_attacking:
                    self.current_state.away_red_cards += 1
                else:
                    self.current_state.home_red_cards += 1
                
                self.current_state.events.append(MatchEvent(
                    minute=self.current_state.minute,
                    event_type=EventType.RED_CARD,
                    team=defending_team.name,
                    description=f"🟥 Red card for {defending_team.name}!"
                ))
    
    def _simulate_match_quick(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile
    ):
        """Quick simulation without detailed events (for large datasets)."""
        
        # Use Poisson-based approach similar to basic simulator
        home_strength = home_profile.attack_strength + home_profile.home_advantage
        away_strength = away_profile.attack_strength
        
        home_xg = 1.4 * (home_strength / 80) * (1 - away_profile.defense_strength / 200)
        away_xg = 1.4 * (away_strength / 80) * (1 - home_profile.defense_strength / 200)
        
        home_xg *= (1 + (home_profile.form - 6.5) * 0.05)
        away_xg *= (1 + (away_profile.form - 6.5) * 0.05)
        
        self.current_state.home_score = int(np.random.poisson(max(0.3, home_xg)))
        self.current_state.away_score = int(np.random.poisson(max(0.3, away_xg)))
        
        # Generate aggregate statistics
        self._generate_aggregate_stats(home_profile, away_profile)
    
    def _generate_aggregate_stats(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile
    ):
        """Generate aggregate match statistics."""
        
        # Possession
        poss_diff = (home_profile.midfield_strength - away_profile.midfield_strength) * 0.3
        self.current_state.home_possession = max(30, min(70, 50 + poss_diff))
        self.current_state.away_possession = 100 - self.current_state.home_possession
        
        # Shots
        self.current_state.home_shots = int(10 + home_profile.attack_strength / 10 + 
                                           self.current_state.home_score * 2 + 
                                           random.uniform(-3, 3))
        self.current_state.away_shots = int(10 + away_profile.attack_strength / 10 + 
                                           self.current_state.away_score * 2 + 
                                           random.uniform(-3, 3))
        
        # Shots on target
        self.current_state.home_shots_on_target = int(
            max(self.current_state.home_score, 
                self.current_state.home_shots * random.uniform(0.35, 0.50))
        )
        self.current_state.away_shots_on_target = int(
            max(self.current_state.away_score,
                self.current_state.away_shots * random.uniform(0.35, 0.50))
        )
        
        # Other stats
        self.current_state.home_corners = int(4 + (self.current_state.home_possession - 50) / 10 + 
                                              random.uniform(-2, 2))
        self.current_state.away_corners = int(4 + (self.current_state.away_possession - 50) / 10 + 
                                              random.uniform(-2, 2))
        
        self.current_state.home_fouls = int(12 - home_profile.discipline / 10 + random.uniform(-2, 2))
        self.current_state.away_fouls = int(12 - away_profile.discipline / 10 + random.uniform(-2, 2))
        
        self.current_state.home_yellow_cards = int(max(0, self.current_state.home_fouls / 6 + 
                                                      random.uniform(-0.5, 1)))
        self.current_state.away_yellow_cards = int(max(0, self.current_state.away_fouls / 6 + 
                                                      random.uniform(-0.5, 1)))
        
        self.current_state.home_red_cards = 1 if random.random() < 0.05 else 0
        self.current_state.away_red_cards = 1 if random.random() < 0.05 else 0
    
    def _calculate_minute_possession(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile,
        home_tactics: TacticalSetup,
        away_tactics: TacticalSetup
    ) -> float:
        """Calculate possession for current minute."""
        
        # Base on midfield quality
        base = 50 + (home_profile.midfield_strength - away_profile.midfield_strength) * 0.3
        
        # Adjust for playing style
        if home_tactics.style == PlayingStyle.POSSESSION:
            base += 10
        elif away_tactics.style == PlayingStyle.POSSESSION:
            base -= 10
        
        if home_tactics.style == PlayingStyle.DEFENSIVE:
            base -= 5
        elif away_tactics.style == PlayingStyle.DEFENSIVE:
            base += 5
        
        # Momentum effect
        base += self.current_state.momentum * 0.1
        
        # Random variation
        base += random.uniform(-5, 5)
        
        return max(30, min(70, base))
    
    def _calculate_line_height(
        self,
        team: AdvancedTeamProfile,
        opponent: AdvancedTeamProfile
    ) -> float:
        """Calculate defensive line height based on team and opponent."""
        
        if team.preferred_style == PlayingStyle.HIGH_PRESS:
            return 65 + team.pressing_ability * 0.2
        elif team.preferred_style == PlayingStyle.DEFENSIVE:
            return 30 + team.defense_strength * 0.1
        else:
            return 50
    
    def _adjust_half_time_tactics(
        self,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile,
        home_tactics: TacticalSetup,
        away_tactics: TacticalSetup
    ):
        """Adjust tactics at half-time based on score."""
        
        score_diff = self.current_state.home_score - self.current_state.away_score
        
        # If losing, become more attacking
        if score_diff < -1:
            home_tactics.line_height += 10
            home_tactics.pressing_intensity += 15
            home_tactics.tempo += 10
        elif score_diff > 1:
            away_tactics.line_height += 10
            away_tactics.pressing_intensity += 15
            away_tactics.tempo += 10
        
        # If winning by 2+, become more defensive
        if score_diff >= 2:
            home_tactics.line_height -= 10
            home_tactics.pressing_intensity -= 10
        elif score_diff <= -2:
            away_tactics.line_height -= 10
            away_tactics.pressing_intensity -= 10
    
    def _create_match_data(
        self,
        home_team: str,
        away_team: str,
        date: str,
        competition: str,
        season: str,
        home_profile: AdvancedTeamProfile,
        away_profile: AdvancedTeamProfile
    ) -> MatchData:
        """Create MatchData object from match state."""
        
        # Calculate halftime scores (roughly 40-45% of goals in first half)
        ht_ratio = random.uniform(0.40, 0.50)
        ht_home = int(self.current_state.home_score * ht_ratio)
        ht_away = int(self.current_state.away_score * ht_ratio)
        
        # Calculate xG
        home_xg = round(self.current_state.home_shots_on_target * random.uniform(0.12, 0.18), 2)
        away_xg = round(self.current_state.away_shots_on_target * random.uniform(0.12, 0.18), 2)
        
        home_stats = TeamStats(
            team_name=home_team,
            goals=self.current_state.home_score,
            shots=self.current_state.home_shots,
            shots_on_target=self.current_state.home_shots_on_target,
            possession=round(self.current_state.home_possession, 1),
            corners=self.current_state.home_corners,
            fouls=self.current_state.home_fouls,
            yellow_cards=self.current_state.home_yellow_cards,
            red_cards=self.current_state.home_red_cards,
            xg=home_xg
        )
        
        away_stats = TeamStats(
            team_name=away_team,
            goals=self.current_state.away_score,
            shots=self.current_state.away_shots,
            shots_on_target=self.current_state.away_shots_on_target,
            possession=round(self.current_state.away_possession, 1),
            corners=self.current_state.away_corners,
            fouls=self.current_state.away_fouls,
            yellow_cards=self.current_state.away_yellow_cards,
            red_cards=self.current_state.away_red_cards,
            xg=away_xg
        )
        
        venue = self._get_venue(home_team)
        attendance = self._simulate_attendance(venue, home_team, away_team)
        
        return MatchData(
            match_id=f"TACTICAL_{season.replace('-', '')}_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}_{date.replace('-', '')}",
            date=date,
            time="15:00",
            competition=competition,
            season=season,
            home_team=home_team,
            away_team=away_team,
            is_arsenal_home=(home_team == "Arsenal"),
            home_score=self.current_state.home_score,
            away_score=self.current_state.away_score,
            halftime_home_score=ht_home,
            halftime_away_score=ht_away,
            venue=venue,
            attendance=attendance,
            home_stats=home_stats,
            away_stats=away_stats
        )
    
    def _get_venue(self, home_team: str) -> str:
        """Get venue for home team."""
        venues = {
            "Arsenal": "Emirates Stadium",
            "Manchester City": "Etihad Stadium",
            "Liverpool": "Anfield",
            "Manchester United": "Old Trafford",
            "Chelsea": "Stamford Bridge",
            "Tottenham": "Tottenham Hotspur Stadium",
            "Newcastle": "St. James' Park",
        }
        return venues.get(home_team, f"{home_team} Stadium")
    
    def _simulate_attendance(self, venue: str, home_team: str, away_team: str) -> int:
        """Simulate attendance."""
        capacities = {
            "Emirates Stadium": 60704,
            "Etihad Stadium": 53400,
            "Anfield": 53394,
            "Old Trafford": 74140,
        }
        capacity = capacities.get(venue, 30000)
        big_teams = ["Arsenal", "Manchester City", "Liverpool", "Manchester United"]
        
        if home_team in big_teams and away_team in big_teams:
            fill_rate = random.uniform(0.95, 0.99)
        else:
            fill_rate = random.uniform(0.85, 0.95)
        
        return int(capacity * fill_rate)
    
    def get_event_log(self) -> List[MatchEvent]:
        """Get the event log from the last simulated match."""
        if self.current_state:
            return self.current_state.events
        return []
    
    def get_match_summary(self) -> str:
        """Get a text summary of the match."""
        if not self.current_state:
            return "No match simulated yet."
        
        events = self.current_state.events
        goals = [e for e in events if e.event_type == EventType.GOAL]
        cards = [e for e in events if e.event_type in [EventType.YELLOW_CARD, EventType.RED_CARD]]
        
        summary = f"\n{'='*60}\n"
        summary += f"MATCH SUMMARY\n"
        summary += f"{'='*60}\n\n"
        summary += f"Final Score: {self.current_state.home_score} - {self.current_state.away_score}\n"
        summary += f"Possession: {self.current_state.home_possession:.1f}% - {self.current_state.away_possession:.1f}%\n"
        summary += f"Shots: {self.current_state.home_shots} - {self.current_state.away_shots}\n"
        summary += f"Shots on Target: {self.current_state.home_shots_on_target} - {self.current_state.away_shots_on_target}\n\n"
        
        if goals:
            summary += "GOALS:\n"
            for goal in goals:
                summary += f"  {goal.minute}' - {goal.team}\n"
            summary += "\n"
        
        if cards:
            summary += "CARDS:\n"
            for card in cards:
                summary += f"  {card.minute}' - {card.description}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("🎮 Tactical Match Simulator\n")
    
    simulator = TacticalMatchSimulator(seed=42)
    
    # Simulate with detailed events
    print("Simulating Arsenal vs Manchester City with full tactical dynamics...\n")
    
    match_data, match_state = simulator.simulate_tactical_match(
        home_team="Arsenal",
        away_team="Manchester City",
        date="2024-03-31",
        detailed_events=True
    )
    
    # Print summary
    print(simulator.get_match_summary())
    
    # Print key events
    print("\n" + "="*60)
    print("KEY EVENTS")
    print("="*60 + "\n")
    
    key_events = [e for e in match_state.events if e.event_type in [
        EventType.GOAL, EventType.RED_CARD, EventType.YELLOW_CARD, 
        EventType.HALF_TIME, EventType.FULL_TIME
    ]]
    
    for event in key_events:
        print(f"{event.minute}' - {event.description}")
