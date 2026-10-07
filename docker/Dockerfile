# TensorFold on NGC PyTorch for GB10 (sm_121, aarch64).
# The CUDA kernels JIT-compile at first start (build.py asks the GPU for its arch), not here;
# run.sh mounts a host cache so only the first boot pays for it.
ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_SHA=cb2ebf0540f42604e2759b2ddef497861e928248
# Optional patch on top of TF_SHA, a file under docker/patches/ (`none` = upstream as tagged).
# patches/flashnext-tools-0.6.6.patch: --tool-system on the CUDA server; a tool call keeps only declared parameters;
# copy drafts; image and video input on two ranks (rank 0 encodes, rank 1 receives the features).
ARG TF_PATCH=none
ARG TF_PATCH_SHA=none
# response_format / guided_* need xgrammar on BOTH ranks: rank 1 recompiles each grammar, and a missing
# xgrammar there raises out of its follow loop and leaves rank 0 hanging in NCCL.
ARG XGRAMMAR=xgrammar==0.2.8
# PyAV decodes video_url parts for the vision tower (VISION=1); without it the server accepts images only.
ARG PYAV=av==19.0.1
COPY patches/ /opt/tensorfold-patches/
# Versions the measured images resolved; a rebuild installs the same ones (base pins come from NGC).
COPY constraints.txt /opt/tensorfold-constraints.txt
# torch and triton come from the base image; fail the build if pip replaces them.
RUN pip freeze | grep -iE '^(torch|triton) @ ' > /opt/base-pins.txt \
 && git clone --quiet --filter=blob:none "${TF_REPO}" /opt/tensorfold-src \
 && git -C /opt/tensorfold-src checkout --quiet "${TF_SHA}" \
 && if [ "${TF_PATCH}" != none ]; then \
      echo "${TF_PATCH_SHA}  /opt/tensorfold-patches/${TF_PATCH#patches/}" | sha256sum -c - \
      && git -C /opt/tensorfold-src apply --index "/opt/tensorfold-patches/${TF_PATCH#patches/}"; fi \
 && env -u HF_TOKEN pip install --no-cache-dir -c /opt/tensorfold-constraints.txt /opt/tensorfold-src "${XGRAMMAR}" "${PYAV}" \
 && pip freeze | grep -iE '^(torch|triton) @ ' | diff - /opt/base-pins.txt \
 && python -c "import sys, tensorfold; assert sys.version_info >= (3, 11); print('tensorfold', tensorfold.__version__)" \
 && python -c "import xgrammar, transformers, av; from transformers import PreTrainedTokenizerFast" \
 && python -c "from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig" \
 && command -v ninja nvcc && test -e /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
 && { echo "${TF_SHA}"; echo "${TF_PATCH} ${TF_PATCH_SHA}"; } > /opt/tensorfold-sha
LABEL tensorfold.sha=${TF_SHA} tensorfold.repo=${TF_REPO} tensorfold.patch=${TF_PATCH} tensorfold.patch_sha=${TF_PATCH_SHA}
ENV PYTHONUNBUFFERED=1 TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1 HF_HOME=/cache/huggingface
