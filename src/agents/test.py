from my_agent import MyAgent
import numpy as np

agent = MyAgent()
# Test with a dummy observation
print(f"Action: {agent.act(np.random.rand(6))}")