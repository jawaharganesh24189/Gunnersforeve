"""
Data preprocessing utilities for football tactics transformer.

This module handles encoding of formations, positions, opposition data,
and tactical situations into formats suitable for the transformer model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class TacticsEncoder:
    """
    Encodes football tactical information into numerical representations.
    """
    
    def __init__(self):
        # Define vocabularies for different tactical elements
        self.formations = {
            '4-4-2': 1,
            '4-3-3': 2,
            '3-5-2': 3,
            '4-2-3-1': 4,
            '3-4-3': 5,
            '5-3-2': 6,
            '4-5-1': 7,
            '4-1-4-1': 8,
            '<PAD>': 0
        }
        
        self.positions = {
            'GK': 1,   # Goalkeeper
            'LB': 2,   # Left Back
            'CB': 3,   # Center Back
            'RB': 4,   # Right Back
            'LWB': 5,  # Left Wing Back
            'RWB': 6,  # Right Wing Back
            'CDM': 7,  # Central Defensive Midfielder
            'CM': 8,   # Central Midfielder
            'LM': 9,   # Left Midfielder
            'RM': 10,  # Right Midfielder
            'CAM': 11, # Central Attacking Midfielder
            'LW': 12,  # Left Winger
            'RW': 13,  # Right Winger
            'ST': 14,  # Striker
            'CF': 15,  # Center Forward
            '<PAD>': 0,
            '<START>': 16,
            '<END>': 17
        }
        
        self.actions = {
            'short_pass': 1,
            'long_pass': 2,
            'through_ball': 3,
            'cross': 4,
            'switch_play': 5,
            'back_pass': 6,
            'forward_pass': 7,
            'diagonal_pass': 8,
            '<PAD>': 0,
            '<START>': 9,
            '<END>': 10
        }
        
        self.tactical_contexts = {
            'counter_attack': 1,
            'possession': 2,
            'high_press': 3,
            'low_block': 4,
            'build_from_back': 5,
            'direct_play': 6,
            '<PAD>': 0
        }
        
        # Inverse mappings for decoding
        self.inv_formations = {v: k for k, v in self.formations.items()}
        self.inv_positions = {v: k for k, v in self.positions.items()}
        self.inv_actions = {v: k for k, v in self.actions.items()}
        self.inv_tactical_contexts = {v: k for k, v in self.tactical_contexts.items()}
    
    def encode_formation(self, formation: str) -> int:
        """Encode formation string to integer"""
        return self.formations.get(formation, self.formations['<PAD>'])
    
    def encode_position(self, position: str) -> int:
        """Encode player position to integer"""
        return self.positions.get(position, self.positions['<PAD>'])
    
    def encode_action(self, action: str) -> int:
        """Encode passing action to integer"""
        return self.actions.get(action, self.actions['<PAD>'])
    
    def encode_tactical_context(self, context: str) -> int:
        """Encode tactical context to integer"""
        return self.tactical_contexts.get(context, self.tactical_contexts['<PAD>'])
    
    def encode_position_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """
        Encode field position coordinates (0-100 for both x and y).
        x: 0 (own goal) to 100 (opponent goal)
        y: 0 (left touchline) to 100 (right touchline)
        """
        x_encoded = int(max(0, min(100, x)))
        y_encoded = int(max(0, min(100, y)))
        return x_encoded, y_encoded
    
    def decode_position(self, position_id: int) -> str:
        """Decode position integer to string"""
        return self.inv_positions.get(position_id, '<UNK>')
    
    def decode_action(self, action_id: int) -> str:
        """Decode action integer to string"""
        return self.inv_actions.get(action_id, '<UNK>')
    
    def decode_formation(self, formation_id: int) -> str:
        """Decode formation integer to string"""
        return self.inv_formations.get(formation_id, '<UNK>')
    
    def encode_tactical_situation(
        self,
        own_formation: str,
        opponent_formation: str,
        ball_position: Tuple[float, float],
        tactical_context: str,
        player_positions: List[Tuple[str, float, float]]
    ) -> np.ndarray:
        """
        Encode a complete tactical situation.
        
        Args:
            own_formation: Team's formation (e.g., '4-3-3')
            opponent_formation: Opponent's formation
            ball_position: (x, y) coordinates of ball
            tactical_context: Current tactical situation
            player_positions: List of (position, x, y) for each player
        
        Returns:
            Encoded array representing the situation
        """
        encoded = []
        
        # Encode formations
        encoded.append(self.encode_formation(own_formation))
        encoded.append(self.encode_formation(opponent_formation))
        
        # Encode ball position
        ball_x, ball_y = self.encode_position_coordinates(ball_position[0], ball_position[1])
        encoded.append(ball_x)
        encoded.append(ball_y)
        
        # Encode tactical context
        encoded.append(self.encode_tactical_context(tactical_context))
        
        # Encode player positions (position type + coordinates)
        for pos, x, y in player_positions:
            encoded.append(self.encode_position(pos))
            pos_x, pos_y = self.encode_position_coordinates(x, y)
            encoded.append(pos_x)
            encoded.append(pos_y)
        
        return np.array(encoded, dtype=np.int32)
    
    def encode_passing_sequence(
        self,
        sequence: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Encode a passing sequence.
        
        Args:
            sequence: List of (position, action) tuples representing the pass sequence
        
        Returns:
            Encoded array
        """
        encoded = [self.actions['<START>']]
        
        for position, action in sequence:
            encoded.append(self.encode_position(position))
            encoded.append(self.encode_action(action))
        
        encoded.append(self.actions['<END>'])
        
        return np.array(encoded, dtype=np.int32)
    
    def decode_passing_sequence(
        self,
        encoded_sequence: np.ndarray
    ) -> List[Tuple[str, str]]:
        """
        Decode an encoded passing sequence.
        
        Args:
            encoded_sequence: Encoded sequence array
        
        Returns:
            List of (position, action) tuples
        """
        sequence = []
        i = 0
        
        while i < len(encoded_sequence):
            if encoded_sequence[i] == self.actions['<START>']:
                i += 1
                continue
            if encoded_sequence[i] == self.actions['<END>']:
                break
            if encoded_sequence[i] == self.actions['<PAD>']:
                i += 1
                continue
            
            # Decode position and action pairs
            if i + 1 < len(encoded_sequence):
                position = self.decode_position(int(encoded_sequence[i]))
                action = self.decode_action(int(encoded_sequence[i + 1]))
                if position != '<PAD>' and action != '<PAD>':
                    sequence.append((position, action))
                i += 2
            else:
                break
        
        return sequence


class TacticsDataset:
    """
    Creates and manages datasets for training the tactics transformer.
    """
    
    def __init__(self, encoder: TacticsEncoder):
        self.encoder = encoder
    
    def create_sample_dataset(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a sample dataset for demonstration/testing.
        In practice, this would load from real match data.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1']
        contexts = ['counter_attack', 'possession', 'build_from_back']
        positions = ['CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'ST']
        actions = ['short_pass', 'long_pass', 'through_ball', 'forward_pass']
        
        input_sequences = []
        target_sequences = []
        
        for _ in range(num_samples):
            # Random tactical situation
            own_formation = np.random.choice(formations)
            opp_formation = np.random.choice(formations)
            ball_pos = (np.random.uniform(10, 30), np.random.uniform(20, 80))
            context = np.random.choice(contexts)
            
            # Random player positions (simplified)
            player_positions = [
                (np.random.choice(positions), 
                 np.random.uniform(0, 100), 
                 np.random.uniform(0, 100))
                for _ in range(5)
            ]
            
            # Encode input
            input_seq = self.encoder.encode_tactical_situation(
                own_formation, opp_formation, ball_pos, context, player_positions
            )
            
            # Random passing sequence (simplified)
            seq_length = np.random.randint(3, 7)
            passing_seq = [
                (np.random.choice(positions), np.random.choice(actions))
                for _ in range(seq_length)
            ]
            
            # Encode target
            target_seq = self.encoder.encode_passing_sequence(passing_seq)
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
        
        # Pad sequences to same length
        max_input_len = max(len(seq) for seq in input_sequences)
        max_target_len = max(len(seq) for seq in target_sequences)
        
        padded_inputs = np.zeros((num_samples, max_input_len), dtype=np.int32)
        padded_targets = np.zeros((num_samples, max_target_len), dtype=np.int32)
        
        for i, (inp, tar) in enumerate(zip(input_sequences, target_sequences)):
            padded_inputs[i, :len(inp)] = inp
            padded_targets[i, :len(tar)] = tar
        
        return padded_inputs, padded_targets


def prepare_training_data(
    num_samples: int = 1000,
    test_split: float = 0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare training and test datasets.
    
    Args:
        num_samples: Total number of samples to generate
        test_split: Fraction of data to use for testing
    
    Returns:
        ((train_inputs, train_targets), (test_inputs, test_targets))
    """
    encoder = TacticsEncoder()
    dataset = TacticsDataset(encoder)
    
    inputs, targets = dataset.create_sample_dataset(num_samples)
    
    # Split into train and test
    split_idx = int(len(inputs) * (1 - test_split))
    
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    test_inputs = inputs[split_idx:]
    test_targets = targets[split_idx:]
    
    return (train_inputs, train_targets), (test_inputs, test_targets)
