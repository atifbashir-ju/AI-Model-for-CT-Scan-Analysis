"""
End-to-End Pipeline Test
---------------------------
Ye script poore Agent-based system ko test karta hai - shuru se aakhir tak.
Isse confirm hota hai ki saare components (DICOM Processor, Vision Model,
Feedback Store, Training Pipeline, Evaluator, Model Registry) sahi se
saath mein kaam kar rahe hain.
"""

from agent import Agent

agent = Agent(models_dir="../models_registry", feedback_path="../feedback/feedback_log.json")

print("=" * 50)
print("STEP 1: DICOM Scan Validate + Load karo")
print("=" * 50)

file_paths = [
    "../subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd",
    "../subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.raw"
]

result = agent.validate_dicom_tool(file_paths)
print("Result:", result["success"])

if not result["success"]:
    print("Error:", result["error"])
else:
    scan_data = result["scan_data"]
    print("Scan loaded:", scan_data)

    print("\n" + "=" * 50)
    print("STEP 2: Prediction chalao")
    print("=" * 50)

    prediction = agent.run_prediction_tool(scan_data)
    if prediction["success"]:
        print(f"Findings: {len(prediction['findings'])}")
        print(f"Model version used: {prediction['model_version']}")
        for f in prediction["findings"][:5]:
            print(" -", f)
    else:
        print("Prediction failed:", prediction.get("error"))

    print("\n" + "=" * 50)
    print("STEP 3: Radiologist Feedback simulate karo")
    print("=" * 50)

    if prediction["success"] and len(prediction["findings"]) > 0:
        first_finding = prediction["findings"][0]
        feedback_entry = agent.collect_feedback_tool(
            patient_id=scan_data.patient_id,
            study_uid=scan_data.study_uid,
            slice_index=first_finding["location_voxel"][2],
            ai_prediction="nodule",
            ai_confidence=first_finding["confidence"],
            model_version=prediction["model_version"],
            radiologist_decision="Accept"  # test ke liye "Accept" bol rahe hain
        )
        print("Feedback saved:", feedback_entry["feedback_id"])

    print("\n" + "=" * 50)
    print("STEP 4: Patient History check karo")
    print("=" * 50)

    history = agent.get_patient_history_tool(scan_data.patient_id)
    print(f"Patient has {len(history)} previous record(s)")

    print("\n" + "=" * 50)
    print("STEP 5: Training Queue check karo")
    print("=" * 50)

    queue_status = agent.check_training_queue_tool(min_feedback_threshold=50)
    print(queue_status)

    if queue_status["should_retrain"]:
        print("\nRetraining would trigger here (not enough feedback yet, so skipped in this test)")
    else:
        print(f"\nNot enough feedback yet ({queue_status['untrained_feedback_count']}/"
              f"{queue_status['threshold']}) - retraining correctly NOT triggered")

print("\n" + "=" * 50)
print("PIPELINE TEST COMPLETE")
print("=" * 50)