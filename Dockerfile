FROM lukemathwalker/cargo-chef:latest-rust-1.85-bookworm AS chef
WORKDIR /usr/src

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

# Donwload, configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

# sccache specific variables
ARG SCCACHE_GHA_ENABLED

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel=2024.0.0-49656 \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" > fakeintel.c && \
    gcc -shared -fPIC -o libfakeintel.so fakeintel.c

COPY --from=planner /usr/src/recipe.json recipe.json

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo chef cook --release --features ort,candle,mkl,static-linking --no-default-features --recipe-path recipe.json && sccache -s

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

FROM builder AS http-builder

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo build --release --bin text-embeddings-router --features ort,candle,mkl,static-linking,http --no-default-features && sccache -s

FROM builder AS grpc-builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY proto proto

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo build --release --bin text-embeddings-router --features ort,candle,mkl,static-linking,grpc --no-default-features && sccache -s

# Helper stage to locate libiomp5.so in builder
# Check MKL installation and common Intel oneAPI paths
FROM builder AS libiomp5-finder
RUN find /opt/intel/oneapi -name "libiomp5.so" -type f 2>/dev/null | head -1 | xargs -I {} sh -c 'if [ -n "{}" ]; then cp {} /libiomp5.so; else touch /libiomp5.so; fi' || touch /libiomp5.so

FROM 090802221799.dkr.ecr.us-west-2.amazonaws.com/chainguard/wolfi:latest-20251204-130502 AS base

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_PRELOAD=/usr/local/libfakeintel.so \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apk update && apk add --no-cache ca-certificates-bundle openssl curl bash bzip2

# Install libstdc++ (latest version, provides required C++ ABI versions)
RUN apk add --no-cache libstdc++

# Copy Intel OpenMP (libiomp5.so)
# Try to get it from the finder stage first, fallback to download
COPY --from=libiomp5-finder /libiomp5.so /tmp/libiomp5.so
RUN if [ -f /tmp/libiomp5.so ] && [ -s /tmp/libiomp5.so ]; then \
        cp /tmp/libiomp5.so /usr/lib/libiomp5.so && \
        rm /tmp/libiomp5.so; \
    else \
        echo "libiomp5.so not found in builder, downloading..." && \
        (curl -Lf https://anaconda.org/conda-forge/libiomp/2023.2.0/download/linux-64/libiomp-2023.2.0-h84fe0f5_0.tar.bz2 -o /tmp/libiomp.tar.bz2 && \
         tar -xjf /tmp/libiomp.tar.bz2 -C /tmp 2>/dev/null && \
         find /tmp -name "libiomp5.so" -type f -exec cp {} /usr/lib/libiomp5.so \; && \
         rm -rf /tmp/libiomp* || \
         curl -Lf https://github.com/intel/intel-extension-for-pytorch/releases/download/v2.1.10+gitd5e0c5a/libiomp5.so -o /usr/lib/libiomp5.so || \
         (echo "ERROR: Could not obtain libiomp5.so from any source" && exit 1)); \
    fi && \
    chmod 755 /usr/lib/libiomp5.so

# Copy OpenMP library from builder stage
# Copy the actual library file (not symlinks) and create necessary symlinks
COPY --from=builder /usr/lib/x86_64-linux-gnu/libomp.so* /usr/local/lib/
# Ensure libomp.so symlink exists
RUN if [ -f /usr/local/lib/libomp.so.5 ] && [ ! -f /usr/local/lib/libomp.so ]; then \
        ln -s libomp.so.5 /usr/local/lib/libomp.so; \
    fi && \
    # Update library cache so dynamic linker can find the libraries
    ldconfig

# Copy a lot of the Intel shared objects because of the mkl_serv_intel_cpu_true patch...
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_def.so.2 /usr/local/lib/libmkl_vml_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2 /usr/local/lib/libmkl_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so.2 /usr/local/lib/libmkl_vml_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx512.so.2 /usr/local/lib/libmkl_vml_avx512.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2 /usr/local/lib/libmkl_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx512.so.2 /usr/local/lib/libmkl_avx512.so.2
COPY --from=builder /usr/src/libfakeintel.so /usr/local/libfakeintel.so

FROM base AS grpc

COPY --from=grpc-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]

FROM base AS http

COPY --from=http-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

# Amazon SageMaker compatible image
FROM http AS sagemaker
COPY --chmod=775 sagemaker-entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Default image
FROM http

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
