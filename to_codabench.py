import torch
import base64
import io
import numpy as np

# 1. Load your trained weights
weights = torch.load("src/agents/dqn_weights.pth", map_location="cpu")

# 2. Convert tensors to numpy arrays
numpy_weights = {k: v.numpy() for k, v in weights.items()}

# 3. Save to a compressed byte buffer
buffer = io.BytesIO()
np.savez_compressed(buffer, **numpy_weights)
buffer.seek(0)

# 4. Generate the new string
encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
print("--- COPY THE STRING BELOW ---")
print(encoded_string)