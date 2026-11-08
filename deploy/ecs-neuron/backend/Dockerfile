# syntax=docker/dockerfile:1.7
# Neuron LLM backend image: neuron_server + Neuron runtime libs + S3-pulling entrypoint.
#
# Build context is the REPO ROOT (the build stage compiles runtime/, neuron/cpp/, glm53/ sources):
#   docker build -f deploy/ecs-neuron/backend/Dockerfile -t behemoth-neuron-server .
# Use a binary already built on the trn instance (neuron/cpp/build.sh OUT srv) instead of compiling here:
#   cp OUT/neuron_server deploy/ecs-neuron/backend/bin/neuron_server
#   docker build -f deploy/ecs-neuron/backend/Dockerfile --build-arg SERVER_STAGE=server-prebuilt -t ... .
# (the prebuilt binary must be built on the same Ubuntu release as BASE_IMAGE, against the same libnrt major).
#
# Neuron packages: pin them to the versions matching the aws-neuronx-dkms driver of the ECS Neuron AMI, e.g.
#   --build-arg NEURON_RUNTIME_VERSION=2.x.y.z-hash --build-arg NEURON_COLLECTIVES_VERSION=2.x.y.z-hash
# (`apt-cache madison aws-neuronx-runtime-lib` lists them). Empty = latest in the repo.

ARG BASE_IMAGE=ubuntu:22.04
ARG SERVER_STAGE=server-build

# ---------------------------------------------------------------- Neuron apt repo + runtime libs (shared base)
FROM ${BASE_IMAGE} AS neuron-base
ARG NEURON_RUNTIME_VERSION=""
ARG NEURON_COLLECTIVES_VERSION=""
ARG NEURON_APT_CODENAME=jammy
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
      | gpg --dearmor -o /usr/share/keyrings/neuron.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${NEURON_APT_CODENAME} main" \
      > /etc/apt/sources.list.d/neuron.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "aws-neuronx-runtime-lib${NEURON_RUNTIME_VERSION:+=${NEURON_RUNTIME_VERSION}}" \
      "aws-neuronx-collectives${NEURON_COLLECTIVES_VERSION:+=${NEURON_COLLECTIVES_VERSION}}" \
 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib

# ---------------------------------------------------------------- build stage: neuron/cpp/build.sh OUT srv
FROM neuron-base AS server-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++-12 libboost-dev libpcre2-dev \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY runtime runtime
COPY neuron/cpp neuron/cpp
COPY glm53 glm53
COPY third_party third_party
RUN test -f /opt/aws/neuron/include/nrt/nrt.h || { echo "libnrt headers missing (aws-neuronx-runtime-lib)"; exit 1; } \
 && bash neuron/cpp/build.sh /out/build srv \
 && mkdir -p /out && cp /out/build/neuron_server /out/neuron_server

# ---------------------------------------------------------------- alternative: binary built on the trn instance
FROM scratch AS server-prebuilt
COPY deploy/ecs-neuron/backend/bin/neuron_server /out/neuron_server

FROM ${SERVER_STAGE} AS server

# ---------------------------------------------------------------- runtime stage
FROM neuron-base AS runtime
ARG S5CMD_VERSION=2.2.2
ARG S5CMD_SHA256=""
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libpcre2-8-0 \
 && curl -fsSL --retry 10 --retry-all-errors --retry-delay 20 -o /tmp/s5cmd.tgz \
      "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
 && if [ -n "${S5CMD_SHA256}" ]; then echo "${S5CMD_SHA256}  /tmp/s5cmd.tgz" | sha256sum -c -; fi \
 && tar -xzf /tmp/s5cmd.tgz -C /usr/local/bin s5cmd && rm /tmp/s5cmd.tgz \
 && apt-get purge -y gnupg && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=server /out/neuron_server /usr/local/bin/neuron_server
COPY deploy/ecs-neuron/backend/entrypoint.sh deploy/ecs-neuron/backend/healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh \
 && ldd /usr/local/bin/neuron_server | (! grep "not found")
ENV PATH=/opt/aws/neuron/bin:${PATH}
EXPOSE 1337
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
