import os
import numpy as np
import pydicom
import SimpleITK as sitk


class DicomLoadError(Exception):
    """Jab bhi file load/validate nahi ho paati, ye error uthta hai."""
    pass


class ScanData:
    """
    Ek loaded, validated scan ka structured result.
    Isi object ko Vision Model aur Agent use karenge.
    """
    def __init__(self, hu_volume, patient_id, study_uid, series_uid, source_format):
        self.hu_volume = hu_volume          # 3D numpy array (HU units mein)
        self.patient_id = patient_id
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.source_format = source_format  # "dicom_series", "mhd", "nifti", "nrrd"
        self.shape = hu_volume.shape

    def __repr__(self):
        return (f"ScanData(patient_id={self.patient_id}, "
                f"shape={self.shape}, format={self.source_format})")


def _sort_key(ds):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        return int(getattr(ds, "InstanceNumber", 0))


def validate_dicom_mixup(dcm_paths):
   
    patient_ids = set()
    study_uids = set()

    for path in dcm_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        patient_ids.add(str(getattr(ds, "PatientID", "UNKNOWN")))
        study_uids.add(str(getattr(ds, "StudyInstanceUID", "UNKNOWN")))

    if len(patient_ids) > 1:
        raise DicomLoadError(
            f"Mixup detected: files belong to {len(patient_ids)} different patients "
            f"({patient_ids}). Please upload only one patient's scan at a time."
        )

    if len(study_uids) > 1:
        raise DicomLoadError(
            f"Mixup detected: files belong to {len(study_uids)} different studies "
            f"({study_uids}). Please upload only one study at a time."
        )

    return patient_ids.pop(), study_uids.pop()


def load_dicom_series(dcm_paths):
    """
    Multiple .dcm files (ek poori series) ko load karke, sahi order mein
    sort karke, ek 3D HU volume mein stack karta hai. Mixup bhi check karta hai.
    """
    if len(dcm_paths) == 0:
        raise DicomLoadError("No DICOM files provided.")

    patient_id, study_uid = validate_dicom_mixup(dcm_paths)

    slices = [pydicom.dcmread(p) for p in dcm_paths]
    slices.sort(key=_sort_key)

    series_uid = str(getattr(slices[0], "SeriesInstanceUID", "UNKNOWN"))

    hu_slices = []
    for ds in slices:
        pixel = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu_slices.append(pixel * slope + intercept)

    hu_volume = np.stack(hu_slices, axis=0)

    return ScanData(hu_volume, patient_id, study_uid, series_uid, "dicom_series")


def load_via_sitk(path, source_format):
    """MHD+RAW, NIfTI, ya NRRD files ke liye - SimpleITK se load karta hai."""
    img = sitk.ReadImage(path)
    hu_volume = sitk.GetArrayFromImage(img).astype(np.float32)

    study_uid = os.path.splitext(os.path.basename(path))[0]

    return ScanData(hu_volume, "UNKNOWN", study_uid, study_uid, source_format)


def load_scan(file_paths):
    """
    Main entry point. file_paths ki list dekh kar automatically decide karta hai
    format kya hai, aur sahi loader use karta hai.

    Returns: ScanData object
    Raises: DicomLoadError agar kuch galat ho (mixup, missing files, etc.)
    """
    dcm_files = [p for p in file_paths if p.lower().endswith(".dcm")]
    mhd_files = [p for p in file_paths if p.lower().endswith(".mhd")]
    raw_files = [p for p in file_paths if p.lower().endswith(".raw")]
    nii_files = [p for p in file_paths if p.lower().endswith((".nii", ".nii.gz"))]
    nrrd_files = [p for p in file_paths if p.lower().endswith(".nrrd")]

    if mhd_files:
        if not raw_files:
            raise DicomLoadError(
                "An .mhd file was provided but its matching .raw file is missing. "
                "Please upload both files together."
            )
        return load_via_sitk(mhd_files[0], "mhd")

    elif dcm_files:
        return load_dicom_series(dcm_files)

    elif nii_files:
        return load_via_sitk(nii_files[0], "nifti")

    elif nrrd_files:
        return load_via_sitk(nrrd_files[0], "nrrd")

    else:
        raise DicomLoadError(
            "No valid scan file found. Upload a DICOM series, or an MHD+RAW / "
            "NIfTI / NRRD volume."
        )


def validate_for_model(scan_data, min_slices=32):
    """
    Check karta hai ki scan Vision Model ke liye usable hai ya nahi
    (kam se kam itni slices honi chahiye).
    """
    if scan_data.shape[0] < min_slices:
        raise DicomLoadError(
            f"This scan only has {scan_data.shape[0]} slice(s). The model needs "
            f"at least {min_slices} slices (a full 3D volume) to detect nodules."
        )
    return True