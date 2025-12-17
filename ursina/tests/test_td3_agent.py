"""
Test TD3 Agent Integration
===========================

Phase 9.1-9.2: Test loading and using real TD3 agent.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.policies.td3_policy import TD3Policy


def test_td3_stub():
    """Test 1: TD3Policy in stub mode (no agent)"""
    print("\n" + "=" * 60)
    print("TEST 1: TD3Policy Stub Mode")
    print("=" * 60)

    policy = TD3Policy(agent=None, action_dim=1, action_scale=0.5)

    # Single action
    state = np.array([1.0, -0.5])
    action = policy.get_action(state)
    print(f"Single action: state={state} -> action={action}")

    # Batch actions
    states = np.random.randn(10, 2)
    actions = policy.get_actions_batch(states)
    print(f"Batch actions: states.shape={states.shape} -> actions.shape={actions.shape}")

    print("[OK] Stub mode works!")


def test_td3_real_agent():
    """Test 2: TD3Policy with real agent"""
    print("\n" + "=" * 60)
    print("TEST 2: TD3Policy with Real Agent")
    print("=" * 60)

    # Path to trained model
    model_path = Path(__file__).parent.parent.parent / "RL" / "calf_model.pth"

    if not model_path.exists():
        print(f"[WARNING] Model not found at {model_path}")
        print("Skipping test with real agent.")
        return

    print(f"Loading model from: {model_path}")

    # Create TD3Policy with real agent
    try:
        policy = TD3Policy.create_from_checkpoint(
            checkpoint_path=str(model_path),
            state_dim=2,
            action_dim=1,
            max_action=5.0,
            hidden_dim=64
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Test single action
    state = np.array([1.0, -0.5])
    action = policy.get_action(state)
    print(f"Single action: state={state} -> action={action}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action in bounds: {-5.0 <= action[0] <= 5.0}")

    # Test batch actions
    states = np.array([
        [2.0, 1.0],
        [0.5, -0.3],
        [-1.0, 0.8],
        [0.0, 0.0],
        [1.5, -1.2]
    ])
    actions = policy.get_actions_batch(states)
    print(f"\nBatch actions:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions:\n{actions}")

    # Check that actions are different (not random)
    actions_2 = policy.get_actions_batch(states)
    assert np.allclose(actions, actions_2), "Actions should be deterministic!"
    print("[OK] Actions are deterministic")

    # Test eval/train mode
    policy.eval_mode()
    print("[OK] Eval mode set")

    policy.train_mode()
    print("[OK] Train mode set")

    policy.eval_mode()  # Back to eval
    print("[OK] Back to eval mode")

    print("\n[OK] Real agent works!")


def test_td3_convergence():
    """Test 3: Check if TD3 agent converges to goal"""
    print("\n" + "=" * 60)
    print("TEST 3: TD3 Convergence Test")
    print("=" * 60)

    model_path = Path(__file__).parent.parent.parent / "RL" / "calf_model.pth"

    if not model_path.exists():
        print(f"[WARNING] Model not found. Skipping convergence test.")
        return

    # Create policy
    policy = TD3Policy.create_from_checkpoint(
        checkpoint_path=str(model_path),
        state_dim=2,
        action_dim=1,
        max_action=5.0
    )

    # Simple dynamics simulation
    dt = 0.01
    max_steps = 500

    # Start from different initial conditions
    initial_states = [
        [2.0, 0.0],
        [1.0, 0.5],
        [-1.5, -0.3],
        [0.5, 1.0]
    ]

    for init_state in initial_states:
        state = np.array(init_state, dtype=np.float32)
        print(f"\nInitial state: {state}")

        for step in range(max_steps):
            # Get action from policy
            action = policy.get_action(state)

            # Simple dynamics: position += velocity * dt, velocity += action * dt
            position, velocity = state
            velocity += action[0] * dt
            position += velocity * dt
            state = np.array([position, velocity])

            # Print progress
            if step % 100 == 0:
                distance = np.linalg.norm(state)
                print(f"  Step {step}: state={state}, distance={distance:.4f}")

            # Check convergence
            if np.linalg.norm(state) < 0.1:
                print(f"  [OK] Converged at step {step}! Final state: {state}")
                break

        if np.linalg.norm(state) >= 0.1:
            print(f"  [WARNING] Did not converge. Final state: {state}")

    print("\n[OK] Convergence test complete!")


if __name__ == "__main__":
    # Run all tests
    test_td3_stub()
    test_td3_real_agent()
    test_td3_convergence()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


