# syntax=docker/dockerfile:1
#
# Dockerfile.ml — GPU training sidecar image for the standalone `hermes-ml`
# service (see docker-compose.yml `ml` service).
#
# It layers a CUDA-enabled PyTorch on top of the already-built `hermes-agent`
# image, so the sidecar shares hermes' tooling, the same /opt/hermes/.venv, and
# (via the ~/.hermes mount) the same profiles + skills + workspace — while ONLY
# this image carries the heavy GPU stack. The always-on gateway image stays
# lean and GPU-free, so a broken CUDA/driver stack can never take it down.
#
# Build (the base must exist first — `gateway` service builds it):
#   docker compose build gateway      # -> image: hermes-agent
#   docker compose build ml           # -> image: hermes-agent-ml
#
# torch cu121 wheels require NVIDIA driver >= 525; this host runs 535 (CUDA
# 12.2 capable), so cu121 is the right, well-supported target for the 2080 Ti.
FROM hermes-agent

# Install into the SAME venv the base image set up so `python3`/`hermes` see
# torch. Runs as root (the base image's final USER), which can write to the
# venv. --no-cache keeps the layer from carrying the wheel cache.
#
# torch is PINNED to the exact local version `2.5.1+cu121`. This box runs
# driver 535 (CUDA 12.2 capable), so we MUST target cu121 — cu124/cu130 wheels
# need driver >= 550 and fail with "driver too old" / cuda_available=False.
# Three flags are load-bearing for getting the cu121 build rather than PyPI's
# default (currently cu124/cu130):
#   --index-strategy unsafe-best-match  : let uv consider the cu121 index for
#       torch even though PyPI (the extra index, needed for torch's nvidia-*
#       and sympy deps) also lists `torch`. Without it uv's default first-index
#       strategy locks onto PyPI and the +cu121 pin becomes unsatisfiable.
#   --exclude-newer 2030-01-01          : override [tool.uv] exclude-newer="7 days"
#       in pyproject.toml. The cu121 wheels carry no upload date, so the 7-day
#       window filters them out and resolution fails.
#   torch==2.5.1+cu121 (exact local ver): a bare ==2.5.1 also matches PyPI's
#       cu124-bundled 2.5.1; the +cu121 suffix forces the driver-535-safe build.
#
# NOTE: torchvision is intentionally omitted — the cu121 index has no
# Python-3.13 (cp313) wheel for it (it stops at cp312), and this image's venv
# is py3.13. A mismatched torchvision would crash at import. If vision support
# is needed, the right fix is a py3.12-based ml image; see docker/Dockerfile.ml
# history / the gpu-train skill notes.
RUN . /opt/hermes/.venv/bin/activate && \
    uv pip install --no-cache \
      --index-strategy unsafe-best-match \
      --exclude-newer 2030-01-01 \
      --index-url https://download.pytorch.org/whl/cu121 \
      --extra-index-url https://pypi.org/simple \
      torch==2.5.1+cu121 && \
    uv pip install --no-cache --exclude-newer 2030-01-01 \
      numpy pandas scikit-learn xgboost lightgbm statsmodels

# New venv files are root-owned; mirror the base image's world-readable policy
# so the remapped runtime `hermes` user (HERMES_UID) can import them.
RUN chmod -R a+rX /opt/hermes/.venv

# Copy the poller explicitly rather than relying on it being in the base image:
# the base `hermes-agent` image bakes source at its own build time and may
# predate this file. This guarantees the ml image always carries the current
# poller. (The compose `command` runs it as the container's main process.)
COPY --chmod=0755 docker/ml_job_poller.py /opt/hermes/docker/ml_job_poller.py

# Reuse the base entrypoint (privilege drop + venv activation + bootstrap).
# The compose `command` overrides what runs after that — here, the GPU poller.
