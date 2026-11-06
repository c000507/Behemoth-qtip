"""Compile the small projection for Trn2 without requiring a Neuron device."""
import os
from pathlib import Path
import shutil
import sys
import tempfile

if len(sys.argv) != 2:
    raise SystemExit("usage: python neuron/tests/compile_trn2.py OUTPUT.neff")
destination = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with tempfile.TemporaryDirectory() as directory:
    # Set before importing NKI. Invoking the standalone kernel compiles it
    # first, then attempts execution. A host without libnrt cannot execute it.
    os.environ["NKI_ARTIFACTS_DIR"] = directory
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ["NKI_LNC1_SHARED_HBM"] = "0"
    import numpy as np
    from qtip_kernel import project_rotated
    from qtip_layout import projection_indices

    args = (np.zeros((2, 256), np.uint16), np.zeros((32, 1024), np.uint32),
            *projection_indices(), np.zeros((3, 64), np.float32))
    try:
        project_rotated(*args)
    except ImportError as exc:
        if "nki.runtime is not available" not in str(exc):
            raise
    artifact = Path(directory) / "kernel.neff"
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise RuntimeError("Neuron compiler did not produce kernel.neff")
    shutil.copyfile(artifact, destination)
    print(f"Trn2 NEFF written to {destination}")
