import torch
import numpy as np
import base64
import io

# Load your weights
weights_path = "src/agents/dqn_weights.pth"
state_dict = torch.load(weights_path, map_location="cpu")

# Convert every tensor to a numpy array
numpy_weights = {k: v.numpy() for k, v in state_dict.items()}

# Save to a compressed bytes buffer
buf = io.BytesIO()
np.savez_compressed(buf, **numpy_weights)

# Encode to Base64
encoded_string = base64.b64encode(buf.getvalue()).decode('utf-8')

print("--- COPY THIS NEW STRING FOR YOUR MY_AGENT.PY ---")
print(encoded_string)