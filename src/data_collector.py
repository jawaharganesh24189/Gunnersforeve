"""
Data collector for Arsenal FC match data from various sources.

This module provides functions to fetch and process match data from different APIs.
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

from data_schema import MatchData, TeamStats, Dataset

# Load environment variables
load_dotenv()


class FootballDataCollector:
    """Collector for football-data.org API."""
    
    BASE_URL = "https://api.football-data.org/v4"
    ARSENAL_TEAM_ID = 57  # Arsenal's team ID in football-data.org
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize collector with API key."""
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        self.headers = {
            "X-Auth-Token": self.api_key
        } if self.api_key else {}
    
    def get_matches(self, season: str = "2023", limit: int = 100) -> List[dict]:
        """
        Fetch Arsenal matches for a given season.
        
        Args:
            season: Season year (e.g., "2023" for 2023-24 season)
            limit: Maximum number of matches to fetch
            
        Returns:
            List of match dictionaries
        """
        if not self.api_key:
            print("Warning: No API key provided. Using mock data.")
            return self._get_mock_data()
        
        url = f"{self.BASE_URL}/teams/{self.ARSENAL_TEAM_ID}/matches"
        params = {
            "season": season,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("matches", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def parse_match(self, match_data: dict) -> MatchData:
        """Parse API match data into MatchData schema."""
        utc_date = match_data.get("utcDate", "")
        date = utc_date.split("T")[0] if utc_date else ""
        time = utc_date.split("T")[1][:5] if "T" in utc_date else None
        
        home_team = match_data.get("homeTeam", {}).get("name", "")
        away_team = match_data.get("awayTeam", {}).get("name", "")
        is_arsenal_home = home_team == "Arsenal FC"
        
        score = match_data.get("score", {})
        full_time = score.get("fullTime", {})
        half_time = score.get("halfTime", {})
        
        return MatchData(
            match_id=str(match_data.get("id")),
            date=date,
            time=time,
            competition=match_data.get("competition", {}).get("name", "Unknown"),
            season=match_data.get("season", {}).get("startDate", "")[:4] + "-" + 
                   match_data.get("season", {}).get("endDate", "")[:4][-2:],
            home_team=home_team,
            away_team=away_team,
            is_arsenal_home=is_arsenal_home,
            home_score=full_time.get("home", 0) or 0,
            away_score=full_time.get("away", 0) or 0,
            halftime_home_score=half_time.get("home"),
            halftime_away_score=half_time.get("away"),
            venue=match_data.get("venue"),
        )
    
    def _get_mock_data(self) -> List[dict]:
        """Return mock match data for demonstration."""
        return [
            {
                "id": 1,
                "utcDate": "2023-08-12T14:00:00Z",
                "competition": {"name": "Premier League"},
                "season": {"startDate": "2023-08-01", "endDate": "2024-05-31"},
                "homeTeam": {"name": "Arsenal FC"},
                "awayTeam": {"name": "Nottingham Forest"},
                "score": {
                    "fullTime": {"home": 2, "away": 1},
                    "halfTime": {"home": 1, "away": 0}
                },
                "venue": "Emirates Stadium"
            },
            {
                "id": 2,
                "utcDate": "2023-08-19T14:00:00Z",
                "competition": {"name": "Premier League"},
                "season": {"startDate": "2023-08-01", "endDate": "2024-05-31"},
                "homeTeam": {"name": "Crystal Palace"},
                "awayTeam": {"name": "Arsenal FC"},
                "score": {
                    "fullTime": {"home": 0, "away": 1},
                    "halfTime": {"home": 0, "away": 0}
                },
                "venue": "Selhurst Park"
            }
        ]


def create_dataset_from_api(api_key: Optional[str] = None, season: str = "2023") -> Dataset:
    """
    Create a dataset from API data.
    
    Args:
        api_key: Optional API key for football-data.org
        season: Season year
        
    Returns:
        Dataset object with match data
    """
    collector = FootballDataCollector(api_key)
    matches_data = collector.get_matches(season=season)
    
    matches = [collector.parse_match(match) for match in matches_data]
    
    return Dataset(
        dataset_name=f"Arsenal Matches {season}-{int(season)+1}",
        description=f"Arsenal FC match data for the {season}-{int(season)+1} season",
        source="football-data.org API (or mock data)",
        last_updated=datetime.now().isoformat(),
        matches=matches
    )


def save_dataset(dataset: Dataset, format: str = "json", output_dir: str = "data/processed"):
    """
    Save dataset to file.
    
    Args:
        dataset: Dataset to save
        format: Output format ('json' or 'csv')
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{dataset.dataset_name.replace(' ', '_')}"
    
    if format == "json":
        filepath = os.path.join(output_dir, f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(dataset.model_dump(), f, indent=2)
        print(f"Dataset saved to {filepath}")
    
    elif format == "csv":
        filepath = os.path.join(output_dir, f"{filename}.csv")
        dataset.to_csv(filepath)
        print(f"Dataset saved to {filepath}")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Example usage
    print("Collecting Arsenal match data...")
    dataset = create_dataset_from_api()
    
    print(f"\nCollected {len(dataset.matches)} matches")
    
    # Save in both formats
    save_dataset(dataset, format="json")
    save_dataset(dataset, format="csv")
    
    print("\nDataset collection complete!")
