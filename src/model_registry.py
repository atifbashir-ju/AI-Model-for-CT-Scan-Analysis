

import os
import json
import shutil
import datetime


class ModelRegistryError(Exception):
    """Registry se related koi bhi problem ho toh ye uthta hai."""
    pass


class ModelRegistry:
    """
    Model versions ko manage karne wali main class.
    Folder structure jo ye banati/manage karti hai:

    models/
        v1/
            model.pth
            metadata.json
        v2/
            model.pth
            metadata.json
        registry.json   <- kaunsa version "active" hai, ye track karta hai
    """

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.registry_file = os.path.join(models_dir, "registry.json")
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(self.registry_file):
            self._save_registry({"active_version": None, "versions": []})

    def _load_registry(self):
        with open(self.registry_file, "r") as f:
            return json.load(f)

    def _save_registry(self, data):
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def list_versions(self):
        """Saare available versions ki list deta hai, unki metadata ke saath."""
        registry = self._load_registry()
        versions = []
        for v in registry["versions"]:
            metadata = self.get_metadata(v)
            versions.append(metadata)
        return versions

    def get_active_version(self):
        """Abhi kaunsa version 'active' (production/live) hai, wo batata hai."""
        registry = self._load_registry()
        return registry["active_version"]

    def get_metadata(self, version):
        """Ek specific version ki metadata (accuracy, date, etc.) deta hai."""
        metadata_path = os.path.join(self.models_dir, version, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ModelRegistryError(f"No metadata found for version '{version}'")
        with open(metadata_path, "r") as f:
            return json.load(f)

    def get_model_path(self, version):
        """Ek specific version ki model.pth file ka path deta hai."""
        model_path = os.path.join(self.models_dir, version, "model.pth")
        if not os.path.exists(model_path):
            raise ModelRegistryError(f"No model file found for version '{version}'")
        return model_path

    def register_new_version(self, version_name, source_model_path, metadata_dict,
                              set_as_active=False):
        """
        Naya trained model ko registry mein add karta hai.
        Purane versions ko kabhi overwrite nahi karta - hamesha naya folder banata hai.
        """
        registry = self._load_registry()

        if version_name in registry["versions"]:
            raise ModelRegistryError(
                f"Version '{version_name}' already exists. Choose a different name."
            )

        version_dir = os.path.join(self.models_dir, version_name)
        os.makedirs(version_dir, exist_ok=True)

        # Model file ko registry ke andar copy karo
        dest_model_path = os.path.join(version_dir, "model.pth")
        shutil.copy(source_model_path, dest_model_path)

        # Metadata save karo
        metadata_dict["version"] = version_name
        metadata_dict.setdefault("registered_at", datetime.datetime.now().isoformat())
        metadata_dict.setdefault("status", "candidate")  # naya model hamesha "candidate" se shuru hota hai

        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        registry["versions"].append(version_name)

        if set_as_active:
            registry["active_version"] = version_name

        self._save_registry(registry)
        return metadata_dict

    def promote_to_active(self, version_name):
        """
        Ek version ko 'active' (production/live) banata hai.
        Ye tabhi call hona chahiye jab evaluation confirm kare naya version behtar hai.
        """
        registry = self._load_registry()

        if version_name not in registry["versions"]:
            raise ModelRegistryError(f"Version '{version_name}' does not exist in registry")

        old_active = registry["active_version"]
        registry["active_version"] = version_name
        self._save_registry(registry)

        # Metadata mein bhi status update karo
        metadata = self.get_metadata(version_name)
        metadata["status"] = "active"
        metadata_path = os.path.join(self.models_dir, version_name, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if old_active:
            old_metadata = self.get_metadata(old_active)
            old_metadata["status"] = "retired"
            old_metadata_path = os.path.join(self.models_dir, old_active, "metadata.json")
            with open(old_metadata_path, "w") as f:
                json.dump(old_metadata, f, indent=2)

        return {"promoted": version_name, "replaced": old_active}

    def compare_versions(self, version_a, version_b):
        """
        Do versions ki metadata side-by-side compare karta hai (accuracy wagera).
        Final decision (promote karna hai ya nahi) Evaluator module lega,
        ye sirf raw comparison data deta hai.
        """
        meta_a = self.get_metadata(version_a)
        meta_b = self.get_metadata(version_b)
        return {
            version_a: {"test_accuracy": meta_a.get("test_accuracy")},
            version_b: {"test_accuracy": meta_b.get("test_accuracy")},
        }