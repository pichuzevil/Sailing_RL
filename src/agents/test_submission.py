# test_submission.py
from my_agent import MyAgent
import numpy as np

try:
    # 1. Test Initialization
    my_agent = MyAgent(state_size=6, action_size=9)
    
    # 2. Test Inference
    dummy_obs = np.random.rand(32774) # Simulating full environment observation
    action = my_agent.act(dummy_obs)
    
    print(f"\n🚀 TEST PASSED!")
    print(f"Action chosen: {action}")
    print("Your 'Fat Agent' is ready for submission.")
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")