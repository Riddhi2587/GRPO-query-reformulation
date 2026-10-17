#!/bin/bash
# Set up a fresh JarvisLabs A100 box for GRPO training: JDK (Terrier), Unsloth + TRL + vLLM stack, project dependencies.
# Install order and pins follow the Kaggle practice run (see requirements-train.txt). Run from the repo root: bash train/setup_jarvis.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== JDK for PyTerrier"
if ! command -v java >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq default-jdk-headless) || sudo sh -c 'apt-get update -qq && apt-get install -y -qq default-jdk-headless'
fi
java -version 2>&1 | head -1

echo "== Python packages"
if [ -z "${VIRTUAL_ENV:-}" ]; then export UV_SYSTEM_PYTHON=1; fi  # install into the box's Python, like the Kaggle notebook
python -m pip install -q --upgrade uv
C="-c requirements-train.txt"
uv pip install -q --upgrade $C vllm unsloth bitsandbytes xformers torchvision triton numpy pillow
# torchao floor depends on the torch version (see the practice notebook)
TORCH_MINOR=$(python -c "import torch; print(torch.__version__.split('+')[0].split('.')[1])")
if [ "$TORCH_MINOR" -ge 10 ]; then TORCHAO="torchao>=0.16.0"; else TORCHAO="torchao>=0.16.0,<0.18.0"; fi
uv pip install -q --no-deps --upgrade "$TORCHAO"
uv pip install -q $C transformers trl trackio "huggingface_hub" datasets pillow
uv pip install -q --no-deps --reinstall-package pillow "pillow==11.3.0"
# project dependencies; the constraints file stops them from moving transformers/TRL
uv pip install -q $C python-terrier ir_datasets ranx pyyaml scipy pandas sentence-transformers

echo "== Versions"
python - <<'EOF'
import importlib.metadata as md, shutil, subprocess
for p in ["torch", "transformers", "trl", "vllm", "unsloth", "trackio", "huggingface_hub", "pillow", "python-terrier", "sentence-transformers"]:
    try: print(f"{p:22s} {md.version(p)}")
    except Exception as e: print(f"{p:22s} MISSING ({e})")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
EOF
echo "== Done. Next: python -m train.grpo --model unsloth/Qwen3-1.7B --max-steps 4 --save-steps 2 --run-name smoke"
