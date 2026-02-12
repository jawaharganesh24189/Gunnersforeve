#!/usr/bin/env python3
"""
Script to fix syntax issues in Jupyter notebooks:
1. Remove incorrect import statements for classes defined in the same notebook
2. Add function calls where needed
3. Fix data loading issues
"""

import json
import sys

def fix_enhanced_tactics_notebook():
    """Fix enhanced_tactics_transformer_notebook.ipynb"""
    print("Fixing enhanced_tactics_transformer_notebook.ipynb...")
    
    with open('enhanced_tactics_transformer_notebook.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Fix cell 15 - Remove incorrect imports
    cell_15 = notebook['cells'][15]
    source = ''.join(cell_15['source'])
    
    # Remove the problematic import lines
    source = source.replace('from .transformer_model import create_tactics_transformer\n', '')
    source = source.replace('from .data_preprocessing import TacticsEncoder\n', '')
    
    # Add a comment explaining the change
    lines = source.split('\n')
    # Find where the imports were and add a comment
    for i, line in enumerate(lines):
        if 'import numpy as np' in line:
            # Insert comment before numpy import
            lines.insert(i, '# Note: TacticsEncoder and create_tactics_transformer are defined in previous cells')
            lines.insert(i+1, '# and are available in the notebook namespace')
            break
    
    cell_15['source'] = '\n'.join(lines).split('\n')
    # Ensure each line is a separate string in the list (as Jupyter expects)
    cell_15['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    
    # Find the last code cell and add a call to demonstrate_inference
    last_code_cell_idx = None
    for i in range(len(notebook['cells']) - 1, -1, -1):
        if notebook['cells'][i]['cell_type'] == 'code':
            last_code_cell_idx = i
            break
    
    # Check if demonstrate_inference is already called
    has_demo_call = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'demonstrate_inference()' in source and 'def demonstrate_inference' not in source:
                has_demo_call = True
                break
    
    if not has_demo_call and last_code_cell_idx:
        # Add a new cell to call demonstrate_inference
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Run inference demonstration\n',
                'demonstrate_inference()'
            ]
        }
        notebook['cells'].append(new_cell)
        print("  Added cell to call demonstrate_inference()")
    
    with open('enhanced_tactics_transformer_notebook.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("  ✓ Fixed enhanced_tactics_transformer_notebook.ipynb")


def fix_arsenal_ml_notebook():
    """Fix arsenal_ml_notebook_standalone.ipynb"""
    print("Fixing arsenal_ml_notebook_standalone.ipynb...")
    
    with open('arsenal_ml_notebook_standalone.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Fix cell 30 - Remove incorrect imports
    cell_30 = notebook['cells'][30]
    source = ''.join(cell_30['source'])
    
    # Remove the problematic import lines
    source = source.replace('from transformer_model import create_tactics_transformer\n', '')
    source = source.replace('from data_preprocessing import TacticsEncoder\n', '')
    
    # Update the comment to reflect the fix
    source = source.replace(
        '# NOTE: These imports refer to classes/functions defined in previous cells\n# The notebook is self-contained - no external files needed\n',
        '# NOTE: TacticsEncoder and create_tactics_transformer are defined in previous cells\n# and are available in the notebook namespace - no imports needed\n'
    )
    
    cell_30['source'] = source.split('\n')
    # Ensure each line is a separate string in the list (as Jupyter expects)
    cell_30['source'] = [line + '\n' for line in cell_30['source'][:-1]] + [cell_30['source'][-1]]
    
    # Check if demonstrate_inference is already called
    has_demo_call = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'demonstrate_inference()' in source and 'def demonstrate_inference' not in source:
                has_demo_call = True
                break
    
    if not has_demo_call:
        # Add a new cell to call demonstrate_inference
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Run inference demonstration\n',
                'demonstrate_inference()'
            ]
        }
        notebook['cells'].append(new_cell)
        print("  Added cell to call demonstrate_inference()")
    
    with open('arsenal_ml_notebook_standalone.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("  ✓ Fixed arsenal_ml_notebook_standalone.ipynb")


def fix_football_tactics_notebook():
    """Fix Football_Tactics_Complete.ipynb - improve data loading"""
    print("Fixing Football_Tactics_Complete.ipynb...")
    
    with open('Football_Tactics_Complete.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find cell 15 with StatsBomb data loading
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'statsbomb' in source.lower() and 'urllib' in source:
                # Improve the data loading logic
                new_source = """# Enhanced data loading with proper player, team, and match data fetching
import json

# Initialize data structures
player_data = {}
team_data = {}
match_data = []

# Load StatsBomb open data for training
try:
    import urllib.request
    
    # Fetch competition data (Premier League)
    competition_url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json'
    with urllib.request.urlopen(competition_url) as response:
        competitions = json.loads(response.read())
    
    # Get Premier League competition
    premier_league = [c for c in competitions if c.get('competition_name') == 'Premier League']
    if premier_league:
        comp_id = premier_league[0]['competition_id']
        season_id = premier_league[0]['season_id']
        
        # Fetch matches for this competition
        matches_url = f'https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp_id}/{season_id}.json'
        with urllib.request.urlopen(matches_url) as response:
            matches = json.loads(response.read())
        
        print(f'✓ Loaded {len(matches)} Premier League matches')
        
        # Extract team data from matches
        for match in matches[:10]:  # Limit to first 10 matches for demo
            home_team = match.get('home_team', {})
            away_team = match.get('away_team', {})
            
            # Store team data
            if home_team.get('home_team_name'):
                team_data[home_team['home_team_name']] = {
                    'id': home_team.get('home_team_id'),
                    'name': home_team['home_team_name']
                }
            if away_team.get('away_team_name'):
                team_data[away_team['away_team_name']] = {
                    'id': away_team.get('away_team_id'),
                    'name': away_team['away_team_name']
                }
            
            # Store match data
            match_data.append({
                'match_id': match.get('match_id'),
                'home_team': home_team.get('home_team_name', 'Unknown'),
                'away_team': away_team.get('away_team_name', 'Unknown'),
                'home_score': match.get('home_score', 0),
                'away_score': match.get('away_score', 0),
                'match_date': match.get('match_date', 'Unknown')
            })
            
            # Fetch lineups for player data
            try:
                match_id = match.get('match_id')
                lineups_url = f'https://raw.githubusercontent.com/statsbomb/open-data/master/data/lineups/{match_id}.json'
                with urllib.request.urlopen(lineups_url) as lineup_response:
                    lineups = json.loads(lineup_response.read())
                    
                    for lineup in lineups:
                        team_name = lineup.get('team_name', 'Unknown')
                        for player in lineup.get('lineup', []):
                            player_id = player.get('player_id')
                            player_name = player.get('player_name')
                            if player_id and player_name:
                                player_data[player_id] = {
                                    'name': player_name,
                                    'team': team_name,
                                    'positions': player.get('positions', [])
                                }
            except:
                pass  # Continue if lineup data not available
        
        print(f'✓ Loaded data for {len(team_data)} teams')
        print(f'✓ Loaded data for {len(player_data)} players')
        print(f'✓ Loaded {len(match_data)} match records')
        
        statsbomb_data = {
            'players': player_data,
            'teams': team_data,
            'matches': match_data
        }
        
except Exception as e:
    print(f'⚠ StatsBomb data not available: {str(e)}')
    print(f'✓ Using simulated data only')
    statsbomb_data = None
"""
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print(f"  Updated cell {i} with improved data loading")
                break
    
    with open('Football_Tactics_Complete.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("  ✓ Fixed Football_Tactics_Complete.ipynb")


def main():
    try:
        fix_enhanced_tactics_notebook()
        fix_arsenal_ml_notebook()
        fix_football_tactics_notebook()
        print("\n✓ All notebooks fixed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error fixing notebooks: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
