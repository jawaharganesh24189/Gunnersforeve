#!/usr/bin/env python3
"""
Script to simulate Arsenal FC matches using AI.

Usage:
    python scripts/simulate_matches.py --matches 38 --team Arsenal
    python scripts/simulate_matches.py --league-round 2024-01-20
    python scripts/simulate_matches.py --matches 20 --format csv --seed 42
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator import FootballMatchSimulator, create_simulated_dataset
from data_collector import save_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Arsenal FC and Premier League matches using AI"
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=20,
        help="Number of matches to simulate (default: 20)"
    )
    parser.add_argument(
        "--team",
        type=str,
        default="Arsenal",
        help="Team to simulate (default: Arsenal)"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2023-24",
        help="Season identifier (default: 2023-24)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for simulated datasets"
    )
    parser.add_argument(
        "--league-round",
        type=str,
        help="Simulate a full league round on specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Display match results after simulation"
    )
    
    args = parser.parse_args()
    
    simulator = FootballMatchSimulator(seed=args.seed)
    
    if args.league_round:
        # Simulate a full league round
        print(f"Simulating Premier League round on {args.league_round}...")
        matches = simulator.simulate_league_round(
            date=args.league_round,
            season=args.season
        )
        
        from data_schema import Dataset
        from datetime import datetime
        
        dataset = Dataset(
            dataset_name=f"Simulated_Premier_League_Round_{args.league_round.replace('-', '')}",
            description=f"AI-simulated Premier League matches for {args.league_round}",
            source="AI Simulation Engine v1.0",
            last_updated=datetime.now().isoformat(),
            matches=matches
        )
        
        print(f"\nSimulated {len(matches)} matches:")
        for match in matches:
            print(f"  {match.home_team} {match.home_score}-{match.away_score} {match.away_team}")
    
    else:
        # Simulate season for specific team
        print(f"Simulating {args.matches} matches for {args.team} ({args.season})...")
        dataset = create_simulated_dataset(
            num_matches=args.matches,
            team=args.team,
            season=args.season,
            seed=args.seed
        )
        
        print(f"\nSimulated {len(dataset.matches)} matches")
        
        if args.show_results:
            print("\nMatch Results:")
            for match in dataset.matches:
                arsenal_score = match.home_score if match.is_arsenal_home else match.away_score
                opponent_score = match.away_score if match.is_arsenal_home else match.home_score
                opponent = match.away_team if match.is_arsenal_home else match.home_team
                venue = "H" if match.is_arsenal_home else "A"
                result = "W" if arsenal_score > opponent_score else ("D" if arsenal_score == opponent_score else "L")
                
                print(f"  {match.date} ({venue}) vs {opponent:20s} {arsenal_score}-{opponent_score} [{result}]")
            
            # Calculate statistics
            arsenal_matches = [m for m in dataset.matches if m.is_arsenal_home or 
                             (m.away_team == args.team or m.home_team == args.team)]
            
            wins = sum(1 for m in arsenal_matches if (
                (m.home_team == args.team and m.home_score > m.away_score) or
                (m.away_team == args.team and m.away_score > m.home_score)
            ))
            draws = sum(1 for m in arsenal_matches if m.home_score == m.away_score)
            losses = len(arsenal_matches) - wins - draws
            
            goals_for = sum(
                m.home_score if m.home_team == args.team else m.away_score 
                for m in arsenal_matches
            )
            goals_against = sum(
                m.away_score if m.home_team == args.team else m.home_score 
                for m in arsenal_matches
            )
            
            print(f"\n{args.team} Season Statistics:")
            print(f"  Record: {wins}W {draws}D {losses}L")
            print(f"  Goals: {goals_for} scored, {goals_against} conceded")
            print(f"  Goal Difference: {goals_for - goals_against:+d}")
            print(f"  Points: {wins * 3 + draws} (from {len(arsenal_matches)} matches)")
    
    # Save dataset
    if args.format in ["json", "both"]:
        save_dataset(dataset, format="json", output_dir=args.output_dir)
    
    if args.format in ["csv", "both"]:
        save_dataset(dataset, format="csv", output_dir=args.output_dir)
    
    print("\nSimulation complete!")
    print(f"\n💡 Tip: Use --seed <number> for reproducible results")
    print(f"💡 Tip: Use --show-results to see detailed match outcomes")


if __name__ == "__main__":
    main()
