# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 直接基于预装 DTK 的基础镜像构建 (DTK 已内置于 BASE_IMAGE, CI 内不再安装 DTK):
#   基础镜像来自镜像总表 ${HARBOR_SITE}/hcu/base/dtk:<DTK版本>-<OS>-py<版本>,
#   由 CI 通过 build-arg 按固定格式拼接传入; 换 DTK 包 = 换基础镜像 tag
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV ROCM_PATH=/opt/dtk

RUN echo "source /opt/dtk/env.sh" >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH' >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

# 内网 pip 源 / 资源服务器地址由 CI 通过 build-arg 注入 (仓库变量)
ARG PYPI_URL
ARG RESOURCE_SERVER_URL

# pip 配置 + das-install
RUN TRUSTED_HOST="${PYPI_URL#*://}" && TRUSTED_HOST="${TRUSTED_HOST%%/*}" && \
    mkdir -p ~/.pip && \
    printf '%s\n' \
    '[global]' \
    "index-url = ${PYPI_URL}/nightly/dtk2604/+simple/" \
    "trusted-host = ${TRUSTED_HOST}" \
    > ~/.pip/pip.conf && \
    pip install --ignore-installed --no-cache-dir blinker && \
    printf '%s\n' \
    '#!/bin/bash' \
    '# usage: das-install <pkg>[==<ver>] <torch_tag>' \
    'pkg=$1 tag=$2 prefix=' \
    'if [[ "$pkg" == *==* ]]; then prefix="${pkg#*==}"; pkg="${pkg%%==*}"; fi' \
    'ver=$(pip index versions "$pkg" 2>/dev/null | grep -oP "${prefix}[^ ()]*${tag}[^ ()]*" | head -1)' \
    '[ -z "$ver" ] && echo "Error: No version for $pkg matching $tag" && exit 1' \
    'echo "Installing $pkg==$ver"' \
    'pip install --no-cache-dir --extra-index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com "$pkg==$ver"' \
    > /usr/local/bin/das-install && chmod +x /usr/local/bin/das-install

COPY requirements-test.txt /tmp/requirements-test.txt

# 安装脚本安装成功后会在 $DTK_DIR/hcu_llvm.info 写入本次完整版本/commit 等记录
RUN wget -q "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh" -O /tmp/hcu_llvm_installer.sh && \
    chmod +x /tmp/hcu_llvm_installer.sh && \
    bash /tmp/hcu_llvm_installer.sh --major 1.0.0 && \
    rm -f /tmp/hcu_llvm_installer.sh

#blas cut / shca / UCX 等均已在基础镜像内就绪 (基础镜像含 blas cut 覆盖与 /opt/ucx),
# 此处不再重复安装 DTK 相关组件

# ---layer--- vLLM 框架依赖
ARG VLLM_VERSION
RUN pip install --no-cache-dir ninja wheel setuptools \
    && pip install --no-cache-dir ray[data,train,tune,serve] -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir amdsmi==1.0.0+630c16a6.dirty \
    && pip install --no-cache-dir cupy \
    && pip install --no-cache-dir apache-tvm-ffi==0.1.11 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir vllm==${VLLM_VERSION} \
    && pip install --no-cache-dir vllm-hcu==${VLLM_VERSION} \
    && pip install --no-cache-dir -r /tmp/requirements-test.txt \
    && pip install --no-cache-dir evalscope==1.9.1 pytest==9.1.1 \
    && pip cache purge

# ---layer--- torch + das 加速库
ARG TORCH_VERSION
RUN TORCH_TAG="torch${TORCH_VERSION//./}" \
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchvision \
    && das-install flash_attn ${TORCH_TAG} \
    && das-install lightop ${TORCH_TAG} \
    && das-install lmslim ${TORCH_TAG} \
    && das-install deepgemm ${TORCH_TAG} \
    && das-install aiter ${TORCH_TAG} \
    && pip install --no-cache-dir mooncake_transfer_engine \
    && das-install deep_ep ${TORCH_TAG} \
    && das-install tilelang ${TORCH_TAG} \
    && das-install boltops ${TORCH_TAG} \
    && das-install causal_conv1d==1.5.4 ${TORCH_TAG} \
    && das-install flash_mla ${TORCH_TAG} \
    && das-install fastsafetensors ${TORCH_TAG} \
    && das-install flash_kda ${TORCH_TAG} \
    && das-install vllm-router ${TORCH_TAG} \
    && das-install nixl ${TORCH_TAG} \
    && pip install --no-cache-dir numpy==1.25.0 \
    && pip install --no-cache-dir ai-dynamo==1.4.1 \
    && pip install --no-cache-dir ai-dynamo-runtime==1.4.1 \
    && pip cache purge
    

# UCX 已在基础镜像内 (/opt/ucx, 路径已在其 ENV 中), 不再重复编译

# 构建完成后移除内网 pip 源, 避免运行时意外拉取内网依赖
RUN rm -rf ~/.pip/pip.conf

# DTK 环境变量: 固化 env.sh 的导出(外加 bashrc 追加的 amdsmi 路径), 使 docker run 直接给命令 / sh /
# k8s 等非交互 bash 场景也能加载; bashrc 保留作兜底(驱动缺失时 env.sh 的 .hyhal 分支只有 source 才命中)。
# 放在文件末尾: 不改动其上方各构建步骤所见的环境。换 DTK 包后对照新 env.sh 同步这里(参考 ciimage「DTK 环境变量」)
ENV HYHAL_PATH=/opt/hyhal
ENV DTKROOT=${ROCM_PATH}
ENV AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"
ENV HIP_PATH=${ROCM_PATH}/hip
ENV MIOPEN_FIND_MODE=3
ENV CMAKE_PREFIX_PATH=/opt/hyhal:$CMAKE_PREFIX_PATH
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hip/bin/hipify:${HYHAL_PATH}/bin:${ROCM_PATH}/opencl/bin:$PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${HYHAL_PATH}/lib:${HYHAL_PATH}/lib64:${ROCM_PATH}/dushmem/lib:${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/dcc/gcvm/lib:${ROCM_PATH}/hip/lib:${ROCM_PATH}/llvm/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=${HYHAL_PATH}/lib:${ROCM_PATH}/opencl/lib:${ROCM_PATH}/dushmem/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/opencl/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${ROCM_PATH}/dcc/gcvm/include:${C_INCLUDE_PATH}:${ROCM_PATH}/include:${HYHAL_PATH}/include:${ROCM_PATH}/llvm/include:${ROCM_PATH}/dushmem/include:${ROCM_PATH}/opencl/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH

# 镜像完整引用由 CI 构建时通过 --build-arg 传入, 注入 ENV 便于容器运行时读取
ARG IMAGE_TAG
ENV IMAGE_TAG=${IMAGE_TAG}
