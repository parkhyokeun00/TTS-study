"""
Project-local storage helpers.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict


PROJECT_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
TMP_DIR = os.path.join(OUTPUTS_DIR, "tmp")
GRADIO_TMP_DIR = os.path.join(OUTPUTS_DIR, "gradio")
HF_HOME_DIR = os.path.join(OUTPUTS_DIR, "hf_home")
HF_HUB_DIR = os.path.join(HF_HOME_DIR, "hub")
HF_XET_DIR = os.path.join(HF_HOME_DIR, "xet")


def ensure_storage_dirs() -> Dict[str, str]:
    paths = {
        "outputs": OUTPUTS_DIR,
        "tmp": TMP_DIR,
        "gradio": GRADIO_TMP_DIR,
        "hf_home": HF_HOME_DIR,
        "hf_hub": HF_HUB_DIR,
        "hf_xet": HF_XET_DIR,
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def configure_runtime_storage() -> Dict[str, str]:
    paths = ensure_storage_dirs()
    os.environ["HF_HOME"] = paths["hf_home"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = paths["hf_hub"]
    os.environ["HF_HUB_CACHE"] = paths["hf_hub"]
    os.environ["HF_XET_CACHE"] = paths["hf_xet"]
    os.environ["GRADIO_TEMP_DIR"] = paths["gradio"]
    os.environ["TMPDIR"] = paths["tmp"]
    os.environ["TMP"] = paths["tmp"]
    os.environ["TEMP"] = paths["tmp"]
    tempfile.tempdir = paths["tmp"]
    return paths
