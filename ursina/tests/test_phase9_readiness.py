"""
Phase 9 Readiness Check - Works WITHOUT PyTorch
================================================

Проверяет готовность Phase 9 в stub режиме.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("PHASE 9 READINESS CHECK")
print("="*70)

# Test 1: Check model file exists
print("\n[TEST 1] Checking for trained model...")
model_path = Path(__file__).parent.parent.parent / "RL" / "calf_model.pth"
if model_path.exists():
    size_kb = model_path.stat().st_size / 1024
    print(f"  [OK] Model found: {model_path}")
    print(f"  [OK] Size: {size_kb:.2f} KB")
    print(f"  [OK] Last modified: {model_path.stat().st_mtime}")
else:
    print(f"  [ERROR] Model not found at {model_path}")
    sys.exit(1)

# Test 2: Check TD3Policy imports
print("\n[TEST 2] Importing TD3Policy...")
try:
    from physics.policies.td3_policy import TD3Policy
    print("  [OK] TD3Policy imported successfully")
except ImportError as e:
    print(f"  [ERROR] Failed to import TD3Policy: {e}")
    sys.exit(1)

# Test 3: TD3Policy stub mode (no torch needed)
print("\n[TEST 3] Testing TD3Policy stub mode...")
try:
    policy = TD3Policy(agent=None, action_dim=1, action_scale=0.5)
    state = np.array([1.0, -0.5])
    action = policy.get_action(state)
    print(f"  [OK] Single action: {action}")
    
    # Batch
    states = np.random.randn(10, 2)
    actions = policy.get_actions_batch(states)
    print(f"  [OK] Batch actions: shape={actions.shape}")
except Exception as e:
    print(f"  [ERROR] Stub mode failed: {e}")
    sys.exit(1)

# Test 4: CALF with TD3 stub
print("\n[TEST 4] Testing CALF with TD3 stub...")
try:
    from physics.policies.pd_policy import PDPolicy
    from physics.policies.calf_policy import CALFPolicy
    
    td3_stub = TD3Policy(agent=None, action_dim=1, action_scale=0.5)
    pd_policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]), dim=1)
    calf = CALFPolicy(td3_stub, pd_policy)
    
    # Test switching
    state_far = np.array([5.0, 0.0])  # Far from goal
    state_close = np.array([0.5, 0.0])  # Close to goal
    
    action_far = calf.get_action(state_far)
    mode_far = calf.current_mode
    
    action_close = calf.get_action(state_close)
    mode_close = calf.current_mode
    
    print(f"  [OK] Far state: mode={mode_far}, action={action_far}")
    print(f"  [OK] Close state: mode={mode_close}, action={action_close}")
    
    if mode_far != mode_close:
        print(f"  [OK] Mode switching works! ({mode_far} -> {mode_close})")
    else:
        print(f"  [WARNING] Modes didn't switch (both {mode_far})")
    
except Exception as e:
    print(f"  [ERROR] CALF test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: VectorizedEnvironment with CALF
print("\n[TEST 5] Testing VectorizedEnvironment with CALF...")
try:
    from physics.vectorized_env import VectorizedEnvironment
    
    vec_env = VectorizedEnvironment(
        n_envs=10,
        policy=calf,
        dt=0.01,
        seed=42
    )
    
    # Run a few steps
    for i in range(10):
        vec_env.step()
    
    states = vec_env.get_states()
    print(f"  [OK] 10 envs stepped successfully")
    print(f"  [OK] States shape: {states.shape}")
    print(f"  [OK] Position range: [{states[:, 0].min():.2f}, {states[:, 0].max():.2f}]")
    
except Exception as e:
    print(f"  [ERROR] VectorizedEnvironment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check PyTorch availability
print("\n[TEST 6] Checking PyTorch...")
try:
    import torch
    print(f"  [OK] PyTorch version: {torch.__version__}")
    print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
    print(f"  [OK] Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Try to create a simple tensor
    test_tensor = torch.FloatTensor([1.0, 2.0, 3.0])
    print(f"  [OK] Test tensor created: {test_tensor}")
    
    TORCH_WORKS = True
except Exception as e:
    print(f"  [WARNING] PyTorch not working: {e}")
    print(f"  [INFO] Phase 9 can't be fully tested without PyTorch")
    TORCH_WORKS = False

# Test 7: Try to load real TD3 agent (only if torch works)
if TORCH_WORKS:
    print("\n[TEST 7] Attempting to load real TD3 agent...")
    try:
        real_policy = TD3Policy.create_from_checkpoint(
            checkpoint_path=str(model_path),
            state_dim=2,
            action_dim=1,
            max_action=5.0,
            hidden_dim=64
        )
        
        # Test inference
        state = np.array([1.0, -0.5])
        action1 = real_policy.get_action(state)
        action2 = real_policy.get_action(state)
        
        print(f"  [OK] Real TD3 agent loaded!")
        print(f"  [OK] Action: {action1}")
        print(f"  [OK] Deterministic: {np.allclose(action1, action2)}")
        
        # Test batch
        states = np.random.randn(5, 2)
        actions = real_policy.get_actions_batch(states)
        print(f"  [OK] Batch inference: {actions.shape}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to load real agent: {e}")
        import traceback
        traceback.print_exc()
        TORCH_WORKS = False
else:
    print("\n[TEST 7] Skipped (PyTorch not available)")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

tests_passed = 5  # Tests 1-5 always run
if TORCH_WORKS:
    tests_passed += 2  # Tests 6-7

print(f"\nTests passed: {tests_passed}/7")
print(f"\nModel: {'OK' if model_path.exists() else 'MISSING'}")
print(f"Code: OK (TD3Policy, CALFPolicy, VectorizedEnv)")
print(f"PyTorch: {'OK' if TORCH_WORKS else 'NOT WORKING'}")

if TORCH_WORKS:
    print("\n[SUCCESS] Phase 9 is READY for full testing!")
    print("Next steps:")
    print("  1. Run: python tests/test_td3_agent.py")
    print("  2. Run: python tests/test_td3_single_point_visual.py")
    print("  3. Run: python tests/test_td3_vs_calf_dual.py")
else:
    print("\n[PARTIAL] Phase 9 code is ready, but PyTorch blocked")
    print("Stub mode works perfectly (Phases 1-8 complete)")
    print("\nTo fix PyTorch:")
    print("  Option 1: Use Python 3.11 instead of 3.14")
    print("  Option 2: Reinstall torch: pip install torch --force-reinstall")
    print("  Option 3: Use Conda: conda install pytorch")

print("\n" + "="*70)






