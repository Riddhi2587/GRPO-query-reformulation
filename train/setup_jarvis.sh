#!/bin/bash
# Set up a GPU box for GRPO training: JDK (Terrier), then the Unsloth + TRL + vLLM stack, then project dependencies.
# Create and ACTIVATE a virtual environment first (this script refuses to install globally), then from the repo root:
#     bash train/setup_jarvis.sh
# Install order follows the Kaggle practice run (see requirements-train.txt). The versions in requirements-train.txt are the tested ones.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no virtual environment is active. Create one (python -m venv .venv-train, or uv venv .venv-train)," >&2
  echo "       run 'source .venv-train/bin/activate', then re-run this script." >&2
  exit 1
fi
echo "== Using virtual environment: $VIRTUAL_ENV ($(python --version))"

UNSLOTH_VERSION=$(sed -n 's/^unsloth==\(.*\)$/\1/p' requirements-train.txt)
VLLM_VERSION=$(sed -n 's/^vllm==\(.*\)$/\1/p' requirements-train.txt)
[ -n "$UNSLOTH_VERSION" ] && [ -n "$VLLM_VERSION" ] || { echo "ERROR: unsloth== / vllm== pins missing in requirements-train.txt" >&2; exit 1; }

echo "== JDK for PyTerrier"
if ! command -v java >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq default-jdk-headless) || sudo sh -c 'apt-get update -qq && apt-get install -y -qq default-jdk-headless'
fi
java -version 2>&1 | head -1

echo "== Python packages"
command -v uv >/dev/null 2>&1 || python -m pip install -q uv
C="-c requirements-train.txt"
# 1) vLLM and Unsloth with explicit pins and WITHOUT the constraints file (as the practice run did): a conflict now fails loudly
#    instead of the resolver quietly falling back to an ancient Unsloth.
uv pip install -q --upgrade "vllm==$VLLM_VERSION" "unsloth==$UNSLOTH_VERSION" bitsandbytes xformers torchvision triton numpy pillow
# 2) torchao: the floor depends on the torch version (see the practice notebook)
TORCH_MINOR=$(python -c "import torch; print(torch.__version__.split('+')[0].split('.')[1])")
if [ "$TORCH_MINOR" -ge 10 ]; then TORCHAO="torchao>=0.16.0"; else TORCHAO="torchao>=0.16.0,<0.18.0"; fi
uv pip install -q --no-deps --upgrade "$TORCHAO"
# 3) force the tested versions of the training stack
uv pip install -q --no-deps $C trl
uv pip install -q $C transformers trackio huggingface_hub datasets pillow
uv pip install -q --no-deps --reinstall-package pillow "pillow==11.3.0"
# 4) project dependencies; the constraints file stops them from moving any pin
uv pip install -q $C python-terrier ir_datasets ranx pyyaml scipy pandas sentence-transformers

echo "== Verifying the install (this fails the script if anything is wrong)"
python - "$UNSLOTH_VERSION" <<'PY'
import importlib.metadata as md, subprocess, sys
want = sys.argv[1]
import unsloth  # first, as Unsloth requires; fails here on a broken install
got = md.version("unsloth")
problems = []
if got != want:
    problems.append(f"unsloth {got} installed, expected {want}")
for mod in ("torch", "trl", "vllm", "sentence_transformers", "pyterrier", "trackio"):
    try:
        __import__(mod)
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot import {mod}: {e}")
import torch
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False")
for p in ("torch", "transformers", "trl", "vllm", "unsloth", "unsloth_zoo", "xformers", "trackio", "huggingface_hub", "pillow",
          "python-terrier", "sentence-transformers"):
    try:
        print(f"{p:22s} {md.version(p)}")
    except Exception:  # noqa: BLE001
        print(f"{p:22s} MISSING")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
if problems:
    print("\nSETUP FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("\nAll checks passed.")
PY

uv pip freeze > requirements-train.lock.txt
echo "== Saved the exact versions to requirements-train.lock.txt"
cat <<'USAGE'

== Done. Smoke test (stops if training fails, because of pipefail and &&):
   set -o pipefail
   python -m train.grpo --model unsloth/Qwen3-1.7B --max-steps 4 --save-steps 2 --run-name smoke --gpu-memory 0.4 2>&1 | tee smoke.log \
     && python -m train.eval_ckpt --run smoke --adapter checkpoint-2 --limit 20 2>&1 | tee eval_smoke.log
USAGE
