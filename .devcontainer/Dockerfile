# Keep this environment on Linux x86_64 even on an Apple Silicon host.
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG NEXTEST_VERSION=0.9.146
ARG KACHE_VERSION=0.28.1
ARG UV_VERSION=0.12.21

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang-18 cmake curl gfortran git \
    libclang-18-dev liblapack-dev libopenblas-dev libssl-dev \
    libmpich-dev mpich pkg-config python3 ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install tools in the image; persisted Cargo/cache volumes do not hide binaries.
ENV RUSTUP_HOME=/opt/rustup
ENV PATH=/opt/rust/bin:/opt/devtools/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && CARGO_HOME=/opt/rust sh /tmp/rustup-init.sh -y --profile minimal \
       --default-toolchain stable --component clippy --component rustfmt --no-modify-path \
    && chmod -R a+rX /opt/rustup /opt/rust \
    && mkdir -p /opt/devtools/bin \
    && curl -fsSL "https://get.nexte.st/${NEXTEST_VERSION}/linux" \
       | tar xz -C /opt/devtools/bin \
    && curl -fsSL "https://github.com/kunobi-ninja/kache/releases/download/v${KACHE_VERSION}/kache-x86_64-unknown-linux-musl.tar.gz" \
       -o /tmp/kache-x86_64-unknown-linux-musl.tar.gz \
    && curl -fsSL "https://github.com/kunobi-ninja/kache/releases/download/v${KACHE_VERSION}/kache-x86_64-unknown-linux-musl.tar.gz.sha256" \
       -o /tmp/kache.sha256 \
    && cd /tmp && sha256sum -c kache.sha256 \
    && tar xzf kache-x86_64-unknown-linux-musl.tar.gz -C /opt/devtools/bin \
    && curl -fsSL "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh \
    && UV_INSTALL_DIR=/opt/devtools/bin UV_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh \
    && rm /tmp/rustup-init.sh /tmp/kache-x86_64-unknown-linux-musl.tar.gz /tmp/kache.sha256 /tmp/uv-install.sh

COPY kache.toml /opt/devcontainer/kache.toml
ENV CARGO_HOME=/home/vscode/.cargo \
    CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/linux-x86_64 \
    RUSTC_WRAPPER=/opt/devtools/bin/kache \
    KACHE_CONFIG=/opt/devcontainer/kache.toml \
    UV_CACHE_DIR=/home/vscode/.cache/mvmc/uv \
    LIBCLANG_PATH=/usr/lib/llvm-18/lib \
    CC=clang-18 CXX=clang++-18 FC=gfortran MPICC=/usr/bin/mpicc \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
USER vscode
