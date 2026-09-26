import os
import json
import numpy as np
import torch
import torch.nn as nn


class SimpleNoduleClassifier(nn.Module):
    """
    3D CNN architecture - CT scan patches mein nodule detect karne ke liye.
    Ye wahi architecture hai jo training ke waqt (Jupyter) use hua tha.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class VisionModelError(Exception):
    """Model loading ya prediction ke dauraan koi problem ho toh ye uthta hai."""
    pass


def load_model_from_registry(models_dir, version):
    """
    Model Registry se ek specific version load karta hai.
    Returns: (model, metadata)
    """
    model_path = os.path.join(models_dir, version, "model.pth")
    metadata_path = os.path.join(models_dir, version, "metadata.json")

    if not os.path.exists(model_path):
        raise VisionModelError(f"Model version '{version}' not found at {model_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model = SimpleNoduleClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, metadata


def sliding_window_predict(model, hu_volume, patch_size=32, stride=24, batch_size=32):
    """
    Poore CT scan volume ko chhote patches mein todke, har patch ko model se
    check karta hai. Jahan model 'nodule' bole, uski location note karta hai.
    """
    findings = []
    z_max, y_max, x_max = hu_volume.shape
    normalized = np.clip(hu_volume, -1000, 400)
    normalized = (normalized + 1000) / 1400.0

    patches_batch = []
    locations_batch = []

    for z in range(0, max(z_max - patch_size, 1), stride):
        for y in range(0, y_max - patch_size, stride):
            for x in range(0, x_max - patch_size, stride):
                patch = normalized[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                if patch.shape == (patch_size, patch_size, patch_size):
                    patches_batch.append(patch)
                    locations_batch.append((z, y, x))

                if len(patches_batch) == batch_size:
                    findings.extend(_run_batch(model, patches_batch, locations_batch, patch_size))
                    patches_batch = []
                    locations_batch = []

    if patches_batch:
        findings.extend(_run_batch(model, patches_batch, locations_batch, patch_size))

    return findings


def _run_batch(model, patches_batch, locations_batch, patch_size):
    results = []
    batch_tensor = torch.tensor(np.array(patches_batch), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)

    for i in range(len(patches_batch)):
        if predictions[i].item() == 1:
            z, y, x = locations_batch[i]
            results.append({
                "location_voxel": (x + patch_size // 2, y + patch_size // 2, z + patch_size // 2),
                "confidence": round(confidences[i].item(), 3)
            })
    return results


def merge_nearby_findings(findings, distance_threshold=48, min_confidence=0.85):
    """
    Overlapping findings ko merge karta hai (jaise ek hi nodule ke around
    kayi patches "positive" bol dein) aur low-confidence findings hata deta hai.
    """
    filtered = [f for f in findings if f["confidence"] >= min_confidence]
    filtered.sort(key=lambda x: x["confidence"], reverse=True)

    merged = []
    used = [False] * len(filtered)

    for i, f in enumerate(filtered):
        if used[i]:
            continue
        merged.append(f)
        used[i] = True
        loc1 = np.array(f["location_voxel"])
        for j in range(i + 1, len(filtered)):
            if used[j]:
                continue
            loc2 = np.array(filtered[j]["location_voxel"])
            if np.linalg.norm(loc1 - loc2) < distance_threshold:
                used[j] = True

    return merged


def predict(model, hu_volume, patch_size=32, stride=24,
            distance_threshold=48, min_confidence=0.85):
    """
    Main entry point: poore scan pe prediction chalata hai aur clean,
    merged findings deta hai.
    """
    raw_findings = sliding_window_predict(model, hu_volume, patch_size, stride)
    clean_findings = merge_nearby_findings(raw_findings, distance_threshold, min_confidence)
    return clean_findings