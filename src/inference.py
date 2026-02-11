"""
Inference script for generating passing tactics using the trained transformer model.

This script demonstrates how to use the trained model to generate passing sequences
for different tactical situations.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .transformer_model import create_tactics_transformer
    from .data_preprocessing import TacticsEncoder
except ImportError:
    # Fall back to direct imports for standalone usage
    from transformer_model import create_tactics_transformer
    from data_preprocessing import TacticsEncoder


class TacticsGenerator:
    """
    Generator class for producing passing tactics using the trained transformer model.
    """
    
    def __init__(self, model, encoder: TacticsEncoder, max_length=50):
        """
        Initialize the tactics generator.
        
        Args:
            model: Trained transformer model
            encoder: TacticsEncoder instance
            max_length: Maximum length of generated sequences
        """
        self.model = model
        self.encoder = encoder
        self.max_length = max_length
    
    def generate_tactics(
        self,
        own_formation: str,
        opponent_formation: str,
        ball_position: tuple,
        tactical_context: str,
        player_positions: list,
        temperature: float = 1.0
    ):
        """
        Generate passing tactics for a given tactical situation.
        
        Args:
            own_formation: Team's formation (e.g., '4-3-3')
            opponent_formation: Opponent's formation
            ball_position: (x, y) coordinates of ball
            tactical_context: Current tactical situation
            player_positions: List of (position, x, y) for each player
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            List of (position, action) tuples representing the passing sequence
        """
        # Encode input situation
        input_seq = self.encoder.encode_tactical_situation(
            own_formation,
            opponent_formation,
            ball_position,
            tactical_context,
            player_positions
        )
        
        # Reshape for model input
        input_seq = input_seq.reshape(1, -1)
        
        # Start with START token
        output_seq = [self.encoder.actions['<START>']]
        
        # Generate sequence token by token
        for _ in range(self.max_length):
            # Prepare decoder input
            dec_input = np.array([output_seq])
            
            # Get predictions
            predictions = self.model((input_seq, dec_input), training=False)
            
            # Get the last token prediction
            predictions = predictions[:, -1, :]
            
            # Apply temperature
            predictions = predictions / temperature
            
            # Sample from distribution
            predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
            
            # Check for END token
            if predicted_id == self.encoder.actions['<END>']:
                break
            
            # Add to output sequence
            output_seq.append(int(predicted_id))
        
        # Decode the sequence
        decoded_seq = self.encoder.decode_passing_sequence(np.array(output_seq))
        
        return decoded_seq
    
    def generate_multiple_tactics(
        self,
        own_formation: str,
        opponent_formation: str,
        ball_position: tuple,
        tactical_context: str,
        player_positions: list,
        num_samples: int = 3,
        temperature: float = 1.0
    ):
        """
        Generate multiple passing tactics options.
        
        Args:
            own_formation: Team's formation
            opponent_formation: Opponent's formation
            ball_position: (x, y) coordinates of ball
            tactical_context: Current tactical situation
            player_positions: List of (position, x, y) for each player
            num_samples: Number of different tactics to generate
            temperature: Sampling temperature
        
        Returns:
            List of passing sequences
        """
        tactics = []
        for _ in range(num_samples):
            tactic = self.generate_tactics(
                own_formation,
                opponent_formation,
                ball_position,
                tactical_context,
                player_positions,
                temperature
            )
            tactics.append(tactic)
        
        return tactics
    
    def generate_tactics_beam_search(
        self,
        own_formation: str,
        opponent_formation: str,
        ball_position: tuple,
        tactical_context: str,
        player_positions: list,
        beam_width: int = 5,
        length_penalty: float = 1.0
    ):
        """
        Generate passing tactics using beam search for better quality.
        
        Beam search explores multiple candidate sequences simultaneously,
        keeping only the top beam_width candidates at each step based on
        their cumulative log probabilities.
        
        Args:
            own_formation: Team's formation (e.g., '4-3-3')
            opponent_formation: Opponent's formation
            ball_position: (x, y) coordinates of ball
            tactical_context: Current tactical situation
            player_positions: List of (position, x, y) for each player
            beam_width: Number of beams to maintain (higher = better quality, slower)
            length_penalty: Penalty for longer sequences (1.0 = no penalty)
        
        Returns:
            List of (position, action) tuples representing the best passing sequence
        """
        # Encode input situation
        input_seq = self.encoder.encode_tactical_situation(
            own_formation,
            opponent_formation,
            ball_position,
            tactical_context,
            player_positions
        )
        
        # Reshape for model input
        input_seq = input_seq.reshape(1, -1)
        
        # Initialize beams: each beam is (sequence, log_prob)
        start_token = self.encoder.actions['<START>']
        end_token = self.encoder.actions['<END>']
        beams = [([start_token], 0.0)]  # (sequence, cumulative_log_prob)
        completed_beams = []
        
        # Generate sequences
        for step in range(self.max_length):
            all_candidates = []
            
            # Expand each beam
            for seq, score in beams:
                # Skip if this beam has ended
                if seq[-1] == end_token:
                    completed_beams.append((seq, score))
                    continue
                
                # Prepare decoder input
                dec_input = np.array([seq])
                
                # Get predictions
                predictions = self.model((input_seq, dec_input), training=False)
                
                # Get the last token prediction
                last_token_logits = predictions[:, -1, :].numpy()[0]
                
                # Convert to log probabilities
                log_probs = tf.nn.log_softmax(last_token_logits).numpy()
                
                # Get top k candidates for this beam
                top_k_indices = np.argsort(log_probs)[-beam_width:]
                
                # Create new candidate beams
                for token_id in top_k_indices:
                    new_seq = seq + [int(token_id)]
                    # Add log probability (negative because we want to maximize)
                    new_score = score + log_probs[token_id]
                    all_candidates.append((new_seq, new_score))
            
            # If no candidates, break
            if not all_candidates:
                break
            
            # Select top beam_width candidates
            # Apply length penalty: score / (length ** length_penalty)
            scored_candidates = []
            for seq, score in all_candidates:
                # Apply length normalization
                normalized_score = score / (len(seq) ** length_penalty)
                scored_candidates.append((seq, score, normalized_score))
            
            # Sort by normalized score and keep top beam_width
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in scored_candidates[:beam_width]]
            
            # Check if all beams have ended
            if all(seq[-1] == end_token for seq, _ in beams):
                completed_beams.extend(beams)
                break
        
        # If we have completed beams, use them; otherwise use current beams
        if completed_beams:
            final_beams = completed_beams
        else:
            final_beams = beams
        
        # Select best sequence based on normalized score
        best_seq = None
        best_score = float('-inf')
        for seq, score in final_beams:
            normalized_score = score / (len(seq) ** length_penalty)
            if normalized_score > best_score:
                best_score = normalized_score
                best_seq = seq
        
        # Decode the best sequence
        if best_seq:
            decoded_seq = self.encoder.decode_passing_sequence(np.array(best_seq))
            return decoded_seq
        else:
            # Fallback to empty sequence
            return []


def load_model_for_inference(
    model_path: str,
    num_layers: int = 4,
    d_model: int = 256,
    num_heads: int = 8,
    dff: int = 512,
    input_vocab_size: int = 1000,
    target_vocab_size: int = 1000,
    max_position_encoding: int = 100,
    dropout_rate: float = 0.1
):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to saved model weights
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward dimension
        input_vocab_size: Input vocabulary size
        target_vocab_size: Target vocabulary size
        max_position_encoding: Maximum sequence length
        dropout_rate: Dropout rate
    
    Returns:
        Loaded model
    """
    model = create_tactics_transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_position_encoding=max_position_encoding,
        dropout_rate=dropout_rate
    )
    
    # Build model by running a forward pass
    dummy_input = np.ones((1, 10), dtype=np.int32)
    dummy_target = np.ones((1, 10), dtype=np.int32)
    _ = model((dummy_input, dummy_target), training=False)
    
    # Load weights
    model.load_weights(model_path)
    
    return model


def demonstrate_inference():
    """
    Demonstrate how to use the model for inference.
    This is a simplified example without loading actual trained weights.
    """
    print("=" * 60)
    print("Tactics Transformer Inference Demonstration")
    print("=" * 60)
    
    # Create encoder
    encoder = TacticsEncoder()
    
    # Create model (in practice, you would load trained weights)
    print("\nCreating model...")
    model = create_tactics_transformer(
        num_layers=2,  # Smaller for demo
        d_model=128,
        num_heads=4,
        dff=256,
        input_vocab_size=200,
        target_vocab_size=50,
        max_position_encoding=100,
        dropout_rate=0.1
    )
    
    # Build model
    dummy_input = np.ones((1, 10), dtype=np.int32)
    dummy_target = np.ones((1, 10), dtype=np.int32)
    _ = model((dummy_input, dummy_target), training=False)
    
    print("Model created successfully!")
    
    # Create generator
    generator = TacticsGenerator(model, encoder, max_length=20)
    
    # Example tactical situation
    print("\n" + "=" * 60)
    print("Example Tactical Situation:")
    print("=" * 60)
    
    own_formation = '4-3-3'
    opponent_formation = '4-4-2'
    ball_position = (20, 50)  # Near own goal, center
    tactical_context = 'build_from_back'
    player_positions = [
        ('GK', 5, 50),
        ('CB', 15, 30),
        ('CB', 15, 70),
        ('CDM', 30, 50),
        ('CM', 40, 40)
    ]
    
    print(f"Own Formation: {own_formation}")
    print(f"Opponent Formation: {opponent_formation}")
    print(f"Ball Position: {ball_position}")
    print(f"Tactical Context: {tactical_context}")
    print(f"Key Player Positions:")
    for pos, x, y in player_positions:
        print(f"  {pos}: ({x}, {y})")
    
    # Generate tactics
    print("\n" + "=" * 60)
    print("Generating Passing Tactics...")
    print("=" * 60)
    
    try:
        tactics = generator.generate_multiple_tactics(
            own_formation,
            opponent_formation,
            ball_position,
            tactical_context,
            player_positions,
            num_samples=3,
            temperature=0.8
        )
        
        print(f"\nGenerated {len(tactics)} tactical options:")
        for i, tactic in enumerate(tactics, 1):
            print(f"\nOption {i}:")
            if len(tactic) > 0:
                for j, (position, action) in enumerate(tactic, 1):
                    print(f"  Step {j}: {position} -> {action}")
            else:
                print("  (Empty sequence generated)")
    
    except Exception as e:
        print(f"\nNote: This is a demonstration with an untrained model.")
        print(f"Expected behavior: Model generates random sequences.")
        print(f"To use in production, train the model first using train.py")
        print(f"\nError details: {e}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)
    print("\nTo train the model and get meaningful predictions:")
    print("1. Run: python src/train.py")
    print("2. Use the trained weights with this inference script")


if __name__ == '__main__':
    demonstrate_inference()
