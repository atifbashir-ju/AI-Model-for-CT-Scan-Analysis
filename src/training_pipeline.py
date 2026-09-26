"""
Training Pipeline Module
---------------------------
Ye module decide karta hai kab retraining zaroori hai (feedback accumulation
ke basis par), aur ek naya "candidate" model train karta hai - jo LUNA16 ke
original data + radiologist ke feedback (corrections) dono se seekhta hai.

IMPORTANT: Ye module kabhi bhi production model ko seedha replace nahi karta.
Ye sirf ek naya CANDIDATE model banata hai. Use "active" banana Evaluator
aur Model Registry ka kaam hai, alag se.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import datetime

from vision_model import SimpleNoduleClassifier


class TrainingPipelineError(Exception):
    """Training ke dauraan koi problem ho toh ye uthta hai."""
    pass


def should_trigger_retraining(feedback_store, min_feedback_threshold=50):
    """
    Decide karta hai ki retraining ka time aa gaya hai ya nahi.

    Manager's rule: "Don't retrain immediately after every wrong prediction."
    Isliye hum ek THRESHOLD rakhte hain - jab tak itna naya, unused feedback
    jama nahi hota, retraining trigger nahi hogi.
    """
    untrained = feedback_store.get_untrained_feedback()
    ready = len(untrained) >= min_feedback_threshold

    return {
        "should_retrain": ready,
        "untrained_feedback_count": len(untrained),
        "threshold": min_feedback_threshold
    }


def feedback_to_training_samples(feedback_entries, patch_extractor_fn):
    """
    Radiologist ke feedback entries ko training-ready patches mein convert karta hai.

    patch_extractor_fn: ek function jo (study_uid, slice_index, location) leke
    original scan se patch nikal sake. Ye caller provide karega, kyunki isko
    scan files tak access chahiye hota hai (jo ye module khud nahi karta -
    single responsibility: ye sirf training logic handle karta hai).
    """
    new_positive_patches = []
    new_negative_patches = []

    for entry in feedback_entries:
        location = entry.get("radiologist_location")
        if location is None:
            continue

        patch = patch_extractor_fn(entry["study_uid"], entry["slice_index"], location)
        if patch is None:
            continue

        # Agar radiologist ne "Reject" kiya, matlab yahan koi problem nahi thi -> negative example
        # Agar radiologist ne "Correct" kiya (nayi location di), matlab wahan genuinely problem hai -> positive example
        if entry["radiologist_decision"] == "Reject":
            new_negative_patches.append(patch)
        elif entry["radiologist_decision"] == "Correct":
            new_positive_patches.append(patch)

    return new_positive_patches, new_negative_patches


def train_candidate_model(X_original, y_original,
                           new_positive_patches=None, new_negative_patches=None,
                           num_epochs=30, batch_size=16, learning_rate=0.001):
    """
    Naya candidate model train karta hai - original LUNA16 data + naya
    radiologist feedback data (agar hai) dono ko combine karke.

    Returns: (trained_model, training_metadata)
    """
    new_positive_patches = new_positive_patches or []
    new_negative_patches = new_negative_patches or []

    # Naya feedback data ko original dataset ke saath jodo
    if new_positive_patches:
        X_new_pos = np.array(new_positive_patches, dtype=np.float32)
        X_original = np.concatenate([X_original, X_new_pos], axis=0)
        y_original = np.concatenate([y_original, np.ones(len(X_new_pos))], axis=0)

    if new_negative_patches:
        X_new_neg = np.array(new_negative_patches, dtype=np.float32)
        X_original = np.concatenate([X_original, X_new_neg], axis=0)
        y_original = np.concatenate([y_original, np.zeros(len(X_new_neg))], axis=0)

    # Normalize + shuffle
    X = np.clip(X_original, -1000, 400)
    X = (X + 1000) / 1400.0

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y_original[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = SimpleNoduleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))

        for i in range(0, X_train_t.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Final evaluation is candidate model ka
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = (test_predicted == y_test_t).float().mean().item()

        correct_nodule = ((test_predicted == 1) & (y_test_t == 1)).sum().item()
        correct_normal = ((test_predicted == 0) & (y_test_t == 0)).sum().item()
        false_positive = ((test_predicted == 1) & (y_test_t == 0)).sum().item()
        false_negative = ((test_predicted == 0) & (y_test_t == 1)).sum().item()

    metadata = {
        "created_at": datetime.datetime.now().isoformat(),
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "new_feedback_positive_added": len(new_positive_patches),
        "new_feedback_negative_added": len(new_negative_patches),
        "test_accuracy": round(test_acc * 100, 2),
        "correctly_identified_nodules": correct_nodule,
        "correctly_identified_normal": correct_normal,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "architecture": "SimpleNoduleClassifier (3D CNN)",
        "patch_size": 32,
        "epochs_trained": num_epochs,
        "status": "candidate"  # kabhi seedha "active" nahi - Evaluator decide karega
    }

    return model, metadata


def save_candidate_to_temp(model, temp_path="temp_candidate_model.pth"):
    """Training ke turant baad model ko ek temp location par save karta hai,
    taaki Model Registry isko baad mein register kar sake."""
    torch.save(model.state_dict(), temp_path)
    return temp_path