

import os
import json
import datetime


class FeedbackStore:


    def __init__(self, storage_path="feedback/feedback_log.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        if not os.path.exists(storage_path):
            self._save_all([])

    def _load_all(self):
        with open(self.storage_path, "r") as f:
            return json.load(f)

    def _save_all(self, data):
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_feedback(self, patient_id, study_uid, slice_index,
                      ai_prediction, ai_confidence, model_version,
                      radiologist_decision, radiologist_location=None,
                      radiologist_note=None):
       
        disagreement = radiologist_decision in ("Reject", "Correct")

        entry = {
            "feedback_id": f"{study_uid}-{slice_index}-{datetime.datetime.now().timestamp()}",
            "timestamp": datetime.datetime.now().isoformat(),
            "patient_id": patient_id,
            "study_uid": study_uid,
            "slice_index": slice_index,
            "ai_prediction": ai_prediction,
            "ai_confidence": ai_confidence,
            "model_version": model_version,
            "radiologist_decision": radiologist_decision,
            "radiologist_location": radiologist_location,
            "radiologist_note": radiologist_note,
            "disagreement_detected": disagreement,
            "eligible_for_training": disagreement,  # sirf disagreements training ke layak hain
            "used_in_training": False  # jab retraining ho jaye, ye True ho jayega
        }

        data = self._load_all()
        data.append(entry)
        self._save_all(data)
        return entry

    def get_patient_history(self, patient_id):
        """
        Manager ka Point 5: same patient ke saare purane findings/feedback laata hai.
        """
        data = self._load_all()
        return [entry for entry in data if entry["patient_id"] == patient_id]

    def get_study_feedback(self, study_uid):
        """Ek specific study ke saare feedback entries laata hai."""
        data = self._load_all()
        return [entry for entry in data if entry["study_uid"] == study_uid]

    def get_untrained_feedback(self):
        """
        Wo saare feedback entries jo abhi tak kisi retraining mein use nahi hue -
        Training Pipeline isko use karega decide karne ke liye "kaafi data jama ho gaya kya".
        """
        data = self._load_all()
        return [entry for entry in data if entry["eligible_for_training"]
                and not entry["used_in_training"]]

    def mark_as_used_in_training(self, feedback_ids):
        """Jab retraining ho jaye, un feedback entries ko 'used' mark karta hai."""
        data = self._load_all()
        for entry in data:
            if entry["feedback_id"] in feedback_ids:
                entry["used_in_training"] = True
        self._save_all(data)

    def get_disagreement_rate(self, model_version=None):
        """
        Ek model version ka 'disagreement rate' nikalta hai - kitni baar
        radiologist ne AI se disagree kiya. Evaluator isko use karega.
        """
        data = self._load_all()
        if model_version:
            data = [e for e in data if e["model_version"] == model_version]

        if len(data) == 0:
            return None

        disagreements = sum(1 for e in data if e["disagreement_detected"])
        return {
            "total_feedback": len(data),
            "disagreements": disagreements,
            "disagreement_rate": round(disagreements / len(data), 3)
        }