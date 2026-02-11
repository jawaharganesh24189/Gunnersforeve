#!/usr/bin/env python3
"""
Script to simulate tactical matches with event-level dynamics.

Usage:
    python scripts/simulate_tactical.py Arsenal "Manchester City" --detailed
    python scripts/simulate_tactical.py Arsenal Chelsea --quick --matches 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tactical_simulator import TacticalMatchSimulator, ADVANCED_PL_TEAMS
from data_schema import Dataset
from data_collector import save_dataset
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser(
        description="Simulate tactical football matches with detailed dynamics"
    )
    parser.add_argument(
        "home_team",
        type=str,
        help="Home team name"
    )
    parser.add_argument(
        "away_team",
        type=str,
        help="Away team name"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Enable detailed event-by-event simulation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick simulation without detailed events (faster for large datasets)"
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=1,
        help="Number of matches to simulate (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Match date (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Show detailed event log"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = TacticalMatchSimulator(seed=args.seed)
    
    # Determine simulation mode
    detailed = args.detailed or (args.matches == 1 and not args.quick)
    
    print(f"\n🎮 Tactical Match Simulator")
    print(f"{'='*60}\n")
    
    if args.matches == 1:
        # Single match
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        
        print(f"Simulating: {args.home_team} vs {args.away_team}")
        print(f"Mode: {'Detailed (minute-by-minute)' if detailed else 'Quick'}")
        print(f"Date: {date}\n")
        
        match_data, match_state = simulator.simulate_tactical_match(
            home_team=args.home_team,
            away_team=args.away_team,
            date=date,
            detailed_events=detailed
        )
        
        # Print summary
        print(simulator.get_match_summary())
        
        # Show events if requested
        if args.show_events and match_state.events:
            print(f"\n{'='*60}")
            print(f"DETAILED EVENT LOG ({len(match_state.events)} events)")
            print(f"{'='*60}\n")
            
            for event in match_state.events:
                print(f"{event.minute:3d}' [{event.event_type.value:15s}] {event.team:20s} - {event.description}")
        
        # Show team profiles
        if args.home_team in ADVANCED_PL_TEAMS:
            home_profile = ADVANCED_PL_TEAMS[args.home_team]
            print(f"\n{'='*60}")
            print(f"{args.home_team} Profile:")
            print(f"  Overall: {home_profile.overall_strength:.1f}")
            print(f"  Attack: {home_profile.attack_strength}, Defense: {home_profile.defense_strength}")
            print(f"  Formation: {home_profile.preferred_formation.value}")
            print(f"  Style: {home_profile.preferred_style.value}")
        
        if args.away_team in ADVANCED_PL_TEAMS:
            away_profile = ADVANCED_PL_TEAMS[args.away_team]
            print(f"\n{args.away_team} Profile:")
            print(f"  Overall: {away_profile.overall_strength:.1f}")
            print(f"  Attack: {away_profile.attack_strength}, Defense: {away_profile.defense_strength}")
            print(f"  Formation: {away_profile.preferred_formation.value}")
            print(f"  Style: {away_profile.preferred_style.value}")
        
        # Save if requested
        if args.save:
            dataset = Dataset(
                dataset_name=f"Tactical_{args.home_team}_vs_{args.away_team}_{date.replace('-', '')}",
                description=f"Tactical simulation: {args.home_team} vs {args.away_team}",
                source="Tactical Simulator v2.0",
                last_updated=datetime.now().isoformat(),
                matches=[match_data]
            )
            save_dataset(dataset, format="json", output_dir="data/processed")
            print(f"\n✅ Match saved to data/processed/")
    
    else:
        # Multiple matches
        print(f"Simulating {args.matches} matches: {args.home_team} vs {args.away_team}")
        print(f"Mode: {'Quick' if not detailed else 'Detailed'}\n")
        
        matches = []
        start_date = datetime.now()
        
        for i in range(args.matches):
            date = (start_date + timedelta(days=i*7)).strftime("%Y-%m-%d")
            
            match_data, match_state = simulator.simulate_tactical_match(
                home_team=args.home_team,
                away_team=args.away_team,
                date=date,
                detailed_events=detailed
            )
            
            matches.append(match_data)
            
            result = "W" if match_data.home_score > match_data.away_score else (
                "D" if match_data.home_score == match_data.away_score else "L"
            )
            print(f"  Match {i+1}: {match_data.home_score}-{match_data.away_score} [{result}]")
        
        # Statistics
        home_wins = sum(1 for m in matches if m.home_score > m.away_score)
        draws = sum(1 for m in matches if m.home_score == m.away_score)
        away_wins = sum(1 for m in matches if m.home_score < m.away_score)
        
        print(f"\nResults: {home_wins}W {draws}D {away_wins}L")
        print(f"Average score: {sum(m.home_score for m in matches)/len(matches):.2f} - {sum(m.away_score for m in matches)/len(matches):.2f}")
        
        # Save if requested
        if args.save:
            dataset = Dataset(
                dataset_name=f"Tactical_{args.home_team}_vs_{args.away_team}_Series",
                description=f"Tactical simulation series: {args.home_team} vs {args.away_team} ({args.matches} matches)",
                source="Tactical Simulator v2.0",
                last_updated=datetime.now().isoformat(),
                matches=matches
            )
            save_dataset(dataset, format="both", output_dir="data/processed")
            print(f"\n✅ Dataset saved to data/processed/")
    
    print()


if __name__ == "__main__":
    main()
