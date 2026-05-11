import os
import sys

def get_dqn_save_path():
    """
    Detects environment and returns the appropriate path for weights.
    """
    if 'google.colab' in sys.modules:
        # Check if Google Drive is mounted
        drive_base = "/content/drive/MyDrive/Sailing_RL"
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(drive_base, exist_ok=True)
            return os.path.join(drive_base, "dqn_weights.pth")
        else:
            print("⚠️ Colab detected but Drive not mounted. Saving to local /content/")
            return "/content/dqn_weights.pth"
    else:
        # Local machine path
        local_path = os.path.join("src", "agents", "dqn_weights.pth")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return local_path