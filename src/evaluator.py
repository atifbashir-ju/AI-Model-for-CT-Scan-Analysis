"""
Evaluator Module
------------------
Ye module naye "candidate" model ko current "active" (production) model se
compare karta hai, aur decide karta hai candidate ko promote karna chahiye
ya reject karna chahiye.

IMPORTANT: Ye module bhi khud kabhi promote nahi karta - sirf ek clear
"recommendation" deta hai. Actual promotion Agent ka decision hoga (Manager
ne bola tha har step mein controlled checkpoints hone chahiye).
"""

import numpy as np
import torch


class EvaluationResult:
    """Ek comparison ka structured result."""
    def __init__(self, candidate_version, production_version,
                 candidate_metrics, production_metrics,
                 recommendation, reasons):
        self.candidate_version = candidate_version
        self.production_version = production_version
        self.candidate_metrics = candidate_metrics
        self.production_metrics = production_metrics
        self.recommendation = recommendation  
        self.reasons = reasons  

    def to_dict(self):
        return {
            "candidate_version": self.candidate_version,
            "production_version": self.production_version,
            "candidate_metrics": self.candidate_metrics,
            "production_metrics": self.production_metrics,
            "recommendation": self.recommendation,
            "reasons": self.reasons
        }

    def __repr__(self):
        return f"EvaluationResult(recommendation={self.recommendation}, reasons={self.reasons})"


def compute_metrics(model, X_test_t, y_test_t):
    """
    Ek model ko test data par evaluate karke saari important metrics nikalta hai.
    Sirf accuracy nahi - sensitivity/recall bhi, jo medical imaging mein
    zyada important hota hai (false negative miss karna zyada risky hai).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)

    correct_nodule = ((predicted == 1) & (y_test_t == 1)).sum().item()
    correct_normal = ((predicted == 0) & (y_test_t == 0)).sum().item()
    false_positive = ((predicted == 1) & (y_test_t == 0)).sum().item()
    false_negative = ((predicted == 0) & (y_test_t == 1)).sum().item()

    total = len(y_test_t)
    accuracy = (correct_nodule + correct_normal) / total

    actual_nodules = correct_nodule + false_negative
    recall = correct_nodule / actual_nodules if actual_nodules > 0 else 0

    predicted_nodules = correct_nodule + false_positive
    precision = correct_nodule / predicted_nodules if predicted_nodules > 0 else 0

    return {
        "accuracy": round(accuracy * 100, 2),
        "recall_sensitivity": round(recall * 100, 2),
        "precision": round(precision * 100, 2),
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "total_test_samples": total
    }


def evaluate_candidate(candidate_model, production_model,
                        X_test_t, y_test_t,
                        candidate_version, production_version,
                        min_accuracy_improvement=0.0,
                        max_recall_drop=2.0):
   
    candidate_metrics = compute_metrics(candidate_model, X_test_t, y_test_t)
    production_metrics = compute_metrics(production_model, X_test_t, y_test_t)

    reasons = []
    passed = True

    accuracy_diff = candidate_metrics["accuracy"] - production_metrics["accuracy"]
    if accuracy_diff < min_accuracy_improvement:
        passed = False
        reasons.append(
            f"Accuracy did not improve enough: candidate={candidate_metrics['accuracy']}%, "
            f"production={production_metrics['accuracy']}% (diff={accuracy_diff:.2f})"
        )
    else:
        reasons.append(
            f"Accuracy check passed: candidate={candidate_metrics['accuracy']}% "
            f"vs production={production_metrics['accuracy']}%"
        )

    recall_diff = production_metrics["recall_sensitivity"] - candidate_metrics["recall_sensitivity"]
    if recall_diff > max_recall_drop:
        passed = False
        reasons.append(
            f"Recall (sensitivity) dropped too much: candidate={candidate_metrics['recall_sensitivity']}%, "
            f"production={production_metrics['recall_sensitivity']}% (drop={recall_diff:.2f}) - "
            f"this risks missing more real nodules, rejecting for safety"
        )
    else:
        reasons.append(
            f"Recall check passed: candidate={candidate_metrics['recall_sensitivity']}% "
            f"vs production={production_metrics['recall_sensitivity']}%"
        )

    recommendation = "promote" if passed else "reject"

    return EvaluationResult(
        candidate_version=candidate_version,
        production_version=production_version,
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics,
        recommendation=recommendation,
        reasons=reasons
    )