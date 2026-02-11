#!/usr/bin/env python3
"""
Script to fetch and process Arsenal FC match data.

Usage:
    python scripts/fetch_data.py --season 2023 --format csv
    python scripts/fetch_data.py --api-key YOUR_API_KEY --season 2023 --format json
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collector import create_dataset_from_api, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Fetch Arsenal FC match data")
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for football-data.org (or set FOOTBALL_DATA_API_KEY env var)"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2023",
        help="Season year (e.g., 2023 for 2023-24 season)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for datasets"
    )
    
    args = parser.parse_args()
    
    print(f"Fetching Arsenal match data for season {args.season}...")
    
    # Create dataset
    dataset = create_dataset_from_api(api_key=args.api_key, season=args.season)
    
    print(f"Collected {len(dataset.matches)} matches")
    
    # Save dataset
    if args.format in ["json", "both"]:
        save_dataset(dataset, format="json", output_dir=args.output_dir)
    
    if args.format in ["csv", "both"]:
        save_dataset(dataset, format="csv", output_dir=args.output_dir)
    
    print("\nData collection complete!")
    print(f"\nTo use a real API key, either:")
    print("  1. Pass --api-key YOUR_KEY")
    print("  2. Set FOOTBALL_DATA_API_KEY environment variable")
    print("  3. Add to .env file: FOOTBALL_DATA_API_KEY=your_key_here")


if __name__ == "__main__":
    main()
