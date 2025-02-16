## Install
cd tools/sam2/
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .

## Download weights (no need for scratch-shared, its not too big)
cd checkpoints && \
./download_ckpts.sh && \
cd ..