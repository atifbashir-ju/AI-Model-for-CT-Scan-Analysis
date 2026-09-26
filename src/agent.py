

import os
import logging
import datetime

from dicom_processor import load_scan, validate_for_model, DicomLoadError
from vision_model import load_model_from_registry, predict
from model_registry import ModelRegistry, ModelRegistryError
from feedback_store import FeedbackStore
from training_pipeline import should_trigger_retraining, train_candidate_model, save_candidate_to_temp
from evaluator import evaluate_candidate


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_activity.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Agent")


class Agent:
    """
    Main orchestrator. Ek instance banao, aur uski methods ("tools") call karo
    jo bhi kaam karwana ho.
    """

    def __init__(self, models_dir="models", feedback_path="feedback/feedback_log.json"):
        self.registry = ModelRegistry(models_dir=models_dir)
        self.feedback_store = FeedbackStore(storage_path=feedback_path)
        logger.info("Agent initialized")

    def validate_dicom_tool(self, file_paths):
        """
        Task: 'Analyze this DICOM'
        Agent's first decision: DICOM Processor ko call karo, mixup/format
        check karwao. Agar fail ho, turant rok do - aage kuch mat karo.
        """
        logger.info(f"Task received: validate and load scan ({len(file_paths)} files)")
        try:
            scan_data = load_scan(file_paths)
            validate_for_model(scan_data)
            logger.info(f"DICOM validation successful: patient={scan_data.patient_id}, "
                        f"study={scan_data.study_uid}, shape={scan_data.shape}")
            return {"success": True, "scan_data": scan_data}
        except DicomLoadError as e:
            logger.warning(f"DICOM validation failed: {e}")
            return {"success": False, "error": str(e)}

    def run_prediction_tool(self, scan_data):
        """
        Task: Vision Model se prediction lena.
        Agent decide karta hai KAUNSA model version use karna hai
        (hamesha "active" version, kabhi candidate nahi).
        """
        active_version = self.registry.get_active_version()
        if active_version is None:
            logger.error("No active model version found in registry")
            return {"success": False, "error": "No active model available"}

        logger.info(f"Selected model version: {active_version}")
        model, metadata = load_model_from_registry(self.registry.models_dir, active_version)

        findings = predict(model, scan_data.hu_volume)

        logger.info(f"Prediction completed: {len(findings)} finding(s) for "
                    f"patient={scan_data.patient_id}, model_version={active_version}")

        return {
            "success": True,
            "findings": findings,
            "model_version": active_version
        }

    def collect_feedback_tool(self, patient_id, study_uid, slice_index,
                               ai_prediction, ai_confidence, model_version,
                               radiologist_decision, radiologist_location=None,
                               radiologist_note=None):
        
        entry = self.feedback_store.add_feedback(
            patient_id, study_uid, slice_index,
            ai_prediction, ai_confidence, model_version,
            radiologist_decision, radiologist_location, radiologist_note
        )

        if entry["disagreement_detected"]:
            logger.info(f"DISAGREEMENT recorded: patient={patient_id}, "
                        f"study={study_uid}, decision={radiologist_decision}")
        else:
            logger.info(f"Agreement recorded: patient={patient_id}, study={study_uid}")

        return entry

    def get_patient_history_tool(self, patient_id):
        """Manager's Point 5: purana history dikhana."""
        history = self.feedback_store.get_patient_history(patient_id)
        logger.info(f"Patient history requested: patient={patient_id}, "
                    f"{len(history)} previous record(s) found")
        return history

    def check_training_queue_tool(self, min_feedback_threshold=50):
        """
        Task: Decide karna hai retraining ka time aaya hai ya nahi.
        Manager's rule: threshold cross hone tak wait karo, turant mat karo.
        """
        status = should_trigger_retraining(self.feedback_store, min_feedback_threshold)
        logger.info(f"Training queue checked: {status['untrained_feedback_count']} "
                    f"unused feedback entries (threshold={min_feedback_threshold})")
        return status

    def trigger_training_tool(self, X_original, y_original):
        """
        Task: Naya candidate model train karna.
        NOTE: Ye seedha production ko nahi badalta - sirf ek candidate banata hai.
        """
        logger.info("Training triggered - building candidate model")

        untrained_feedback = self.feedback_store.get_untrained_feedback()

        model, metadata = train_candidate_model(X_original, y_original)

        temp_path = save_candidate_to_temp(model)

        logger.info(f"Candidate model trained: test_accuracy={metadata['test_accuracy']}%")

        return {
            "model": model,
            "metadata": metadata,
            "temp_path": temp_path,
            "feedback_used": [e["feedback_id"] for e in untrained_feedback]
        }

    def evaluate_model_tool(self, candidate_model, X_test_t, y_test_t):
        """
        Task: Candidate ko production se compare karna.
        Agent yahan decide NAHI karta - sirf Evaluator se recommendation leta hai.
        """
        active_version = self.registry.get_active_version()
        production_model, _ = load_model_from_registry(self.registry.models_dir, active_version)

        result = evaluate_candidate(
            candidate_model, production_model, X_test_t, y_test_t,
            candidate_version="candidate", production_version=active_version
        )

        logger.info(f"Evaluation completed: recommendation={result.recommendation}")
        for reason in result.reasons:
            logger.info(f"  Reason: {reason}")

        return result

    def promote_model_tool(self, candidate_temp_path, candidate_metadata,
                            evaluation_result, feedback_ids_used,
                            new_version_name=None, require_human_approval=True):
        """
        Task: Candidate ko production banana - LEKIN sirf tab jab:
        1. Evaluator ne 'promote' recommend kiya ho
        2. (Optional) Human approval mila ho - Manager ne isse strongly suggest kiya
        """
        if evaluation_result.recommendation != "promote":
            logger.warning(f"Promotion blocked: evaluation recommended '{evaluation_result.recommendation}'")
            return {"success": False, "reason": "Evaluation did not recommend promotion"}

        if require_human_approval:
            logger.info("Promotion requires human approval - waiting for confirmation "
                        "(call this again with require_human_approval=False after human reviews)")
            return {
                "success": False,
                "reason": "Awaiting human approval",
                "evaluation_summary": evaluation_result.to_dict()
            }

        if new_version_name is None:
            existing_versions = self.registry.list_versions()
            new_version_name = f"v{len(existing_versions) + 1}"

        self.registry.register_new_version(
            new_version_name, candidate_temp_path, candidate_metadata
        )
        result = self.registry.promote_to_active(new_version_name)

        self.feedback_store.mark_as_used_in_training(feedback_ids_used)

        logger.info(f"Model promoted: {new_version_name} is now active "
                    f"(replaced {result['replaced']})")

        return {"success": True, "new_active_version": new_version_name,
                "replaced_version": result["replaced"]}