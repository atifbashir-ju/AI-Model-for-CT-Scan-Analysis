
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from agent import Agent
from dicom_processor import DicomLoadError

st.set_page_config(page_title="CT Scan Analyzer - Agent System", layout="wide")
st.title("CT Scan Nodule Detection - AI Agent System")
st.write("Upload a CT scan. The AI Agent will validate it, run detection, "
         "and let you review and correct its findings.")

@st.cache_resource
def get_agent():
    return Agent(models_dir="../models_registry", feedback_path="../feedback/feedback_log.json")

agent = get_agent()

with st.sidebar:
    st.header("System Status")
    active_version = agent.registry.get_active_version()
    st.write(f"**Active model version:** {active_version}")

    if active_version:
        metadata = agent.registry.get_metadata(active_version)
        st.write(f"Test accuracy: {metadata.get('test_accuracy')}%")
        st.write(f"Trained on: {metadata.get('total_scans_used')} scans")

    st.divider()
    st.subheader("Training Queue")
    queue_status = agent.check_training_queue_tool(min_feedback_threshold=50)
    st.write(f"Unused feedback: {queue_status['untrained_feedback_count']} / "
             f"{queue_status['threshold']}")
    if queue_status["should_retrain"]:
        st.success("Enough feedback collected - ready for retraining")
    else:
        st.info("Not enough feedback yet for retraining")

uploaded_files = st.file_uploader(
    "Upload CT scan files",
    type=["dcm", "mhd", "raw", "nii", "gz", "nrrd"],
    accept_multiple_files=True
)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uf in uploaded_files:
        save_path = os.path.join(temp_dir, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.read())
        file_paths.append(save_path)

    result = agent.validate_dicom_tool(file_paths)

    if not result["success"]:
        st.error(result["error"])
    else:
        scan_data = result["scan_data"]
        st.success(f"Scan loaded: Patient={scan_data.patient_id}, "
                   f"Study={scan_data.study_uid}, Shape={scan_data.shape}")

        history = agent.get_patient_history_tool(scan_data.patient_id)
        if history:
            with st.expander(f"Patient History - {len(history)} previous record(s)"):
                for h in history:
                    st.write(f"- {h['timestamp'][:10]}: "
                             f"AI said '{h['ai_prediction']}' (conf={h['ai_confidence']}), "
                             f"Radiologist: {h['radiologist_decision']}"
                             + (f" - \"{h['radiologist_note']}\"" if h.get('radiologist_note') else ""))

        cache_key = f"prediction_{scan_data.study_uid}"
        if cache_key not in st.session_state:
            with st.spinner("Agent is running detection on the full scan... this may take a few minutes"):
                prediction = agent.run_prediction_tool(scan_data)
                st.session_state[cache_key] = prediction

        prediction = st.session_state[cache_key]

        if not prediction["success"]:
            st.error(prediction.get("error", "Prediction failed"))
        else:
            findings = prediction["findings"]
            model_version = prediction["model_version"]
            st.write(f"### AI found {len(findings)} suspicious area(s) "
                     f"(model version: {model_version})")

            if findings:
                findings_by_slice = {}
                for f in findings:
                    z = f["location_voxel"][2]
                    findings_by_slice.setdefault(z, []).append(f)

                slice_idx = st.slider(
                    "Navigate scan slices",
                    min_value=0,
                    max_value=scan_data.shape[0] - 1,
                    value=list(findings_by_slice.keys())[0]
                )

                slice_img = scan_data.hu_volume[slice_idx]
                low, high = 40 - 350/2, 40 + 350/2
                windowed = np.clip(slice_img, low, high)
                windowed_disp = ((windowed - low) / (high - low) * 255).astype(np.uint8)
                display_img = cv2.cvtColor(windowed_disp, cv2.COLOR_GRAY2BGR)

                ai_findings_here = findings_by_slice.get(slice_idx, [])
                for f in ai_findings_here:
                    x, y, z = f["location_voxel"]
                    cv2.circle(display_img, (x, y), 15, (255, 0, 0), 2)
                    cv2.putText(display_img, f"{f['confidence']}", (x-15, y-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                st.write(f"AI findings on this slice: {len(ai_findings_here)}")

                st.write("### Radiologist Review")
                st.write("Click on the image to mark the correct location, if different from the AI's mark.")

                marks_key = f"marks_{scan_data.study_uid}_{slice_idx}"
                if marks_key not in st.session_state:
                    st.session_state[marks_key] = []

                marked_img = display_img.copy()
                for mx, my in st.session_state[marks_key]:
                    cv2.drawMarker(marked_img, (mx, my), (0, 255, 0),
                                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                pil_bg = Image.fromarray(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
                click_result = streamlit_image_coordinates(pil_bg, key=f"click_{slice_idx}")

                if click_result is not None:
                    point = (click_result["x"], click_result["y"])
                    if point not in st.session_state[marks_key]:
                        st.session_state[marks_key].append(point)
                        st.rerun()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"Marks placed: {len(st.session_state[marks_key])}")
                with col_b:
                    if st.button("Clear marks", key=f"clear_{slice_idx}"):
                        st.session_state[marks_key] = []
                        st.rerun()

                decision = st.radio(
                    "Radiologist decision for this slice",
                    ["Accept (AI is correct)", "Reject (no problem here)", "Correct (mark shown above)"],
                    key=f"decision_{slice_idx}"
                )

                note = st.text_area("Describe the problem / reasoning", key=f"note_{slice_idx}")

                if st.button("Submit Feedback", key=f"submit_{slice_idx}"):
                    decision_map = {
                        "Accept (AI is correct)": "Accept",
                        "Reject (no problem here)": "Reject",
                        "Correct (mark shown above)": "Correct"
                    }
                    radiologist_decision = decision_map[decision]

                    ai_conf = ai_findings_here[0]["confidence"] if ai_findings_here else None

                    entry = agent.collect_feedback_tool(
                        patient_id=scan_data.patient_id,
                        study_uid=scan_data.study_uid,
                        slice_index=slice_idx,
                        ai_prediction="nodule" if ai_findings_here else "normal",
                        ai_confidence=ai_conf,
                        model_version=model_version,
                        radiologist_decision=radiologist_decision,
                        radiologist_location=st.session_state[marks_key] if st.session_state[marks_key] else None,
                        radiologist_note=note
                    )

                    st.success(f"Feedback saved (disagreement: {entry['disagreement_detected']})")
                    st.session_state[marks_key] = []
            else:
                st.write("No suspicious areas detected.")