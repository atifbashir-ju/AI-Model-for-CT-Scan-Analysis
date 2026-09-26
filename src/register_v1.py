"""
One-time script: Jupyter mein jo model (v1) train kiya tha, use is naye
Model Registry system mein properly register karta hai.
"""

import json
import os
from model_registry import ModelRegistry

# Jupyter wale folder ka path - jaha models/v1/model.pth aur metadata.json hai
JUPYTER_MODELS_PATH = "../models/v1"

metadata_path = os.path.join(JUPYTER_MODELS_PATH, "metadata.json")
model_path = os.path.join(JUPYTER_MODELS_PATH, "model.pth")

if not os.path.exists(metadata_path) or not os.path.exists(model_path):
    print(f"ERROR: Files not found at {JUPYTER_MODELS_PATH}")
    print("Please check the path - it should point to where Jupyter saved models/v1/")
else:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    registry = ModelRegistry(models_dir="../models_registry")

    result = registry.register_new_version(
        version_name="v1",
        source_model_path=model_path,
        metadata_dict=metadata,
        set_as_active=True
    )

    print("Successfully registered v1 in the Model Registry!")
    print(json.dumps(result, indent=2))

    print("\nActive version:", registry.get_active_version())
    print("All versions:", registry.list_versions())