ARG CUDA_DEVEL_IMAGE=docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG CUDA_RUNTIME_IMAGE=docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

FROM ${CUDA_DEVEL_IMAGE} AS builder

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG APT_SECURITY_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
ARG REQUIRE_LOCAL_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN sed -i -E "s|https?://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g; s|https?://security.ubuntu.com/ubuntu|https://${APT_SECURITY_MIRROR}/ubuntu|g" /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        python3.10-venv \
        build-essential \
        git \
        cmake \
        ninja-build \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/pip \
    && printf "[global]\nindex-url = %s\ntrusted-host = %s\n" "$PIP_INDEX_URL" "$PIP_TRUSTED_HOST" > /etc/pip.conf \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app/SegAnyGAussians

COPY requirements.txt ./requirements.txt
COPY submodules ./submodules
COPY third_party ./third_party

RUN test -f requirements.txt \
    && test -f submodules/simple-knn/setup.py \
    && test -f submodules/diff-gaussian-rasterization/setup.py \
    && test -f submodules/diff-gaussian-rasterization-depth/setup.py \
    && test -f submodules/diff-gaussian-rasterization-max-contributor/setup.py \
    && test -f submodules/diff-gaussian-rasterization_contrastive_f/setup.py \
    && test -f third_party/GroundingDINO/setup.py \
    && test -f third_party/segment-anything/setup.py

RUN python -m pip install \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    torchaudio==2.4.1+cu118 \
    --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}"

RUN python -m pip install --no-build-isolation -r requirements.txt \
    && python -m pip install --no-build-isolation ./third_party/GroundingDINO

FROM ${CUDA_RUNTIME_IMAGE}

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG APT_SECURITY_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG REQUIRE_LOCAL_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN sed -i -E "s|https?://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g; s|https?://security.ubuntu.com/ubuntu|https://${APT_SECURITY_MIRROR}/ubuntu|g" /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN mkdir -p /etc/pip \
    && printf "[global]\nindex-url = %s\ntrusted-host = %s\n" "$PIP_INDEX_URL" "$PIP_TRUSTED_HOST" > /etc/pip.conf

WORKDIR /app/SegAnyGAussians

COPY . /app/SegAnyGAussians
COPY --from=builder /opt/venv /opt/venv

RUN if [ "$REQUIRE_LOCAL_WEIGHTS" = "1" ]; then \
        test -f weights/sam_vit_h_4b8939.pth \
        && test -f weights/groundingdino_swint_ogc.pth \
        && test -f weights/GroundingDINO_SwinT_OGC.py; \
    fi

CMD ["bash"]
