"""Run kernel and full-model comparisons on whichever GPU provider was built.

Returns CTest skip code 77 without a usable GPU. Full models are tiny fixtures.
"""
import argparse
import os
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=pathlib.Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="4")
    test = args.build / "glm53_gpu_tests"
    result = subprocess.run([str(test)], env=env)
    if result.returncode:
        return result.returncode
    with tempfile.TemporaryDirectory(prefix="glm53-gpu-model-") as temp:
        root = pathlib.Path(temp)
        for kind, flag, ranks in (("moe", "--fixture", ()), ("dense", "--dense-fixture", ()),
                                  ("tp-moe", "--tp-fixture", (2, 4)), ("tp-dense", "--tp-dense-fixture", (2, 4))):
            source, quant = root / kind, root / (kind + "-nvfp4")
            subprocess.run([str(args.build / "glm53_tests"), flag, str(source)], check=True, env=env)
            subprocess.run([str(args.build / "glm53"), "quantize", "--model", str(source),
                            "--out", str(quant)], check=True, env=env)
            for checkpoint in (source, quant):
                subprocess.run([str(test), "--model", str(checkpoint)], check=True, env=env)
                # Tensor parallelism against the CPU reference and the single-GPU path; ranks share one
                # GPU when fewer are present (GLM53_TP_SAME_DEVICE is set by the test itself).
                for tp in ranks:
                    subprocess.run([str(test), "--model", str(checkpoint), "--tp", str(tp)], check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
