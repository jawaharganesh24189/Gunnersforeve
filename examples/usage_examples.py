"""
Example usage of the Tactics Transformer model.

This script demonstrates how to:
1. Create and configure the model
2. Prepare tactical data
3. Train the model (simplified)
4. Generate passing tactics for different scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from transformer_model import create_tactics_transformer
from data_preprocessing import TacticsEncoder, prepare_training_data
from inference import TacticsGenerator


def example_model_creation():
    """Example: Creating a tactics transformer model"""
    print("=" * 70)
    print("Example 1: Creating a Tactics Transformer Model")
    print("=" * 70)
    
    model = create_tactics_transformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        input_vocab_size=200,
        target_vocab_size=50,
        max_position_encoding=100,
        dropout_rate=0.1
    )
    
    print("\nModel created successfully!")
    print(f"Model type: {type(model).__name__}")
    print("\nModel configuration:")
    print(f"  - Number of layers: 4")
    print(f"  - Model dimension: 256")
    print(f"  - Number of attention heads: 8")
    print(f"  - Feed-forward dimension: 512")
    
    return model


def example_data_encoding():
    """Example: Encoding tactical situations"""
    print("\n" + "=" * 70)
    print("Example 2: Encoding Tactical Situations")
    print("=" * 70)
    
    encoder = TacticsEncoder()
    
    # Example 1: Counter-attack from defense
    print("\nScenario 1: Counter-Attack")
    print("-" * 40)
    
    own_formation = '4-3-3'
    opponent_formation = '4-4-2'
    ball_position = (25, 45)  # Just past own third
    tactical_context = 'counter_attack'
    player_positions = [
        ('GK', 5, 50),
        ('CB', 20, 35),
        ('CDM', 35, 50),
        ('CAM', 60, 50),
        ('ST', 80, 50)
    ]
    
    encoded = encoder.encode_tactical_situation(
        own_formation,
        opponent_formation,
        ball_position,
        tactical_context,
        player_positions
    )
    
    print(f"Formation: {own_formation} vs {opponent_formation}")
    print(f"Ball at: {ball_position}")
    print(f"Context: {tactical_context}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"First 10 values: {encoded[:10]}")
    
    # Example 2: Build-up from back
    print("\nScenario 2: Build from Back")
    print("-" * 40)
    
    tactical_context = 'build_from_back'
    ball_position = (15, 50)
    
    encoded2 = encoder.encode_tactical_situation(
        own_formation,
        opponent_formation,
        ball_position,
        tactical_context,
        player_positions
    )
    
    print(f"Context: {tactical_context}")
    print(f"Ball at: {ball_position}")
    print(f"Encoded shape: {encoded2.shape}")
    
    # Example passing sequence
    print("\nExample Passing Sequence Encoding:")
    print("-" * 40)
    
    passing_sequence = [
        ('CB', 'short_pass'),
        ('CDM', 'forward_pass'),
        ('CAM', 'through_ball'),
        ('ST', 'forward_pass')
    ]
    
    encoded_seq = encoder.encode_passing_sequence(passing_sequence)
    print("Original sequence:")
    for pos, action in passing_sequence:
        print(f"  {pos} -> {action}")
    
    print(f"\nEncoded sequence: {encoded_seq}")
    
    # Decode back
    decoded_seq = encoder.decode_passing_sequence(encoded_seq)
    print("\nDecoded sequence:")
    for pos, action in decoded_seq:
        print(f"  {pos} -> {action}")
    
    return encoder


def example_data_preparation():
    """Example: Preparing training data"""
    print("\n" + "=" * 70)
    print("Example 3: Preparing Training Data")
    print("=" * 70)
    
    (train_inputs, train_targets), (test_inputs, test_targets) = prepare_training_data(
        num_samples=100,  # Small sample for demo
        test_split=0.2
    )
    
    print(f"\nTraining set:")
    print(f"  - Input shape: {train_inputs.shape}")
    print(f"  - Target shape: {train_targets.shape}")
    print(f"  - Number of samples: {len(train_inputs)}")
    
    print(f"\nTest set:")
    print(f"  - Input shape: {test_inputs.shape}")
    print(f"  - Target shape: {test_targets.shape}")
    print(f"  - Number of samples: {len(test_inputs)}")
    
    print(f"\nSample input (first 10 values): {train_inputs[0][:10]}")
    print(f"Sample target (first 10 values): {train_targets[0][:10]}")


def example_tactical_scenarios():
    """Example: Different tactical scenarios"""
    print("\n" + "=" * 70)
    print("Example 4: Different Tactical Scenarios")
    print("=" * 70)
    
    encoder = TacticsEncoder()
    
    scenarios = [
        {
            'name': 'High Press Counter',
            'own_formation': '4-3-3',
            'opponent_formation': '4-4-2',
            'ball_position': (70, 50),
            'tactical_context': 'counter_attack',
            'player_positions': [
                ('CB', 20, 50),
                ('CDM', 45, 50),
                ('CAM', 70, 50),
                ('LW', 85, 20),
                ('RW', 85, 80),
                ('ST', 90, 50)
            ]
        },
        {
            'name': 'Possession Build-Up',
            'own_formation': '4-2-3-1',
            'opponent_formation': '5-3-2',
            'ball_position': (30, 50),
            'tactical_context': 'possession',
            'player_positions': [
                ('GK', 5, 50),
                ('LB', 25, 15),
                ('CB', 20, 40),
                ('CB', 20, 60),
                ('RB', 25, 85),
                ('CDM', 40, 50)
            ]
        },
        {
            'name': 'Wing Play',
            'own_formation': '3-5-2',
            'opponent_formation': '4-3-3',
            'ball_position': (50, 20),
            'tactical_context': 'possession',
            'player_positions': [
                ('LWB', 50, 10),
                ('CM', 60, 40),
                ('CAM', 75, 50),
                ('RWB', 50, 90),
                ('ST', 85, 45)
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 40)
        print(f"Formation: {scenario['own_formation']} vs {scenario['opponent_formation']}")
        print(f"Ball Position: {scenario['ball_position']}")
        print(f"Tactical Context: {scenario['tactical_context']}")
        print(f"Key Players:")
        for pos, x, y in scenario['player_positions']:
            print(f"  {pos}: ({x:.0f}, {y:.0f})")


def example_model_inference():
    """Example: Using model for inference (demonstration only)"""
    print("\n" + "=" * 70)
    print("Example 5: Model Inference (Demonstration)")
    print("=" * 70)
    
    print("\nNote: This demonstrates the inference API structure.")
    print("For actual predictions, train the model first using train.py")
    
    # Create encoder
    encoder = TacticsEncoder()
    
    # Create a small model for demonstration
    model = create_tactics_transformer(
        num_layers=2,
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
    
    print("\nModel built successfully!")
    
    # Create generator
    generator = TacticsGenerator(model, encoder, max_length=20)
    
    print("\nGenerator ready for inference.")
    print("\nExample usage:")
    print("```python")
    print("tactics = generator.generate_tactics(")
    print("    own_formation='4-3-3',")
    print("    opponent_formation='4-4-2',")
    print("    ball_position=(25, 50),")
    print("    tactical_context='counter_attack',")
    print("    player_positions=[...]")
    print(")")
    print("```")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("TACTICS TRANSFORMER - EXAMPLE USAGE")
    print("=" * 70)
    print("\nThis script demonstrates the capabilities of the Tactics Transformer")
    print("for generating football passing tactics.\n")
    
    try:
        # Run examples
        example_model_creation()
        example_data_encoding()
        example_data_preparation()
        example_tactical_scenarios()
        example_model_inference()
        
        print("\n" + "=" * 70)
        print("All Examples Completed Successfully!")
        print("=" * 70)
        
        print("\nNext Steps:")
        print("1. Train the model: python src/train.py")
        print("2. Use for inference: python src/inference.py")
        print("3. Customize for your specific needs")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
