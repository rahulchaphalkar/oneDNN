#!/usr/bin/env bash
#*******************************************************************************
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

# Local Linux equivalent of the "CI CPU Windows AMX Debug" job. Run this on an
# AMX-capable Linux machine (SPR/EMR/GNR) to reproduce the same build + targeted
# AMX bf16 tests that produced the illegal-instruction failures on windows-2025.
#
# Steps, mirroring the CI job:
#   1. Build & run the CPU ISA diagnostic (CPUID + XCR0 + arch_prctl + guarded
#      AMX execution probe).
#   2. Configure & build only the targets needed for the AMX bf16 tests.
#   3. Run the same ctest filter with ONEDNN_VERBOSE=1.
#
# Usage:
#   .github/automation/x64/run_amx_debug_linux.sh [build_dir]
#
# Environment overrides:
#   CC, CXX            compilers (default: clang/clang++ if present, else gcc/g++)
#   BUILD_DIR          build directory (default: build-amx-debug)
#   JOBS               parallel build jobs (default: nproc)
#   ONEDNN_MAX_CPU_ISA optional ISA cap, e.g. AVX512_CORE_BF16 to bypass AMX

set -euo pipefail

# Resolve repo root from this script's location (.github/automation/x64/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${1:-${BUILD_DIR:-build-amx-debug}}"
JOBS="${JOBS:-$(nproc)}"

# Pick a CMake generator: prefer Ninja if available, else Unix Makefiles.
if [[ -z "${GENERATOR:-}" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        GENERATOR="Ninja"
    else
        GENERATOR="Unix Makefiles"
    fi
fi

# Pick compilers: prefer clang to match the Windows job, fall back to gcc.
if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
    if command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
        CC="${CC:-clang}"
        CXX="${CXX:-clang++}"
    else
        CC="${CC:-gcc}"
        CXX="${CXX:-g++}"
    fi
fi
export CC CXX

echo "=================================================================="
echo " oneDNN local AMX debug run (Linux)"
echo "=================================================================="
echo "Repo root  : ${REPO_ROOT}"
echo "Build dir  : ${BUILD_DIR}"
echo "Generator  : ${GENERATOR}"
echo "Compilers  : CC=${CC} CXX=${CXX}"
echo "Jobs       : ${JOBS}"
echo "MAX_CPU_ISA: ${ONEDNN_MAX_CPU_ISA:-<unset>}"
echo

# --- Step 1: CPU ISA diagnostic ------------------------------------------------
echo "------------------------------------------------------------------"
echo " Step 1: CPU ISA diagnostic (AMX / BF16)"
echo "------------------------------------------------------------------"
DIAG_SRC="${SCRIPT_DIR}/cpu_isa_check_linux.cpp"
DIAG_BIN="$(mktemp -d)/cpu_isa_check_linux"
# Compile with AMX intrinsics enabled so the probe can exercise the full
# ldtilecfg + tileloadd + tdpbf16ps path (all runtime-guarded against #UD).
# Fall back to a TILERELEASE-only probe if the compiler lacks AMX support.
if "${CXX}" -O2 -std=c++17 -mamx-tile -mamx-bf16 -o "${DIAG_BIN}" "${DIAG_SRC}" 2>/dev/null; then
    :
else
    echo "note: compiler lacks -mamx-tile/-mamx-bf16; building TILERELEASE-only probe"
    "${CXX}" -O2 -std=c++17 -o "${DIAG_BIN}" "${DIAG_SRC}"
fi
"${DIAG_BIN}" || true
echo

# --- Step 2: Configure & build -------------------------------------------------
echo "------------------------------------------------------------------"
echo " Step 2: Configure & build (dnnl test_internals benchdnn)"
echo "------------------------------------------------------------------"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_BUILD_FOR_CI=ON \
    -DDNNL_WERROR=ON \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_GPU_RUNTIME=NONE \
    -DDNNL_TEST_SET=SMOKE \
    -DDNNL_BUILD_EXAMPLES=OFF \
    -DONEDNN_BUILD_GRAPH=OFF \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}"

cmake --build "${BUILD_DIR}" --parallel "${JOBS}" \
    --target dnnl test_internals test_internals_env_vars_dnnl \
    test_internals_env_vars_onednn benchdnn
echo

# --- Step 3: Isolate the first small-tail AMX BF16 BRGEMM case -----------------
echo "------------------------------------------------------------------"
echo " Step 3: Isolate AMX BF16 BRGEMM M=N=K=4 with TILECFG tracing"
echo "------------------------------------------------------------------"
set +e
GTEST_FILTER="TestBRGEMMSimple/brgemm_test_t.TestsBRGEMM/54" \
ONEDNN_TEST_BRGEMM_AMX_TRACE=1 \
ONEDNN_VERBOSE=1 \
ctest --test-dir "${BUILD_DIR}" --verbose --output-on-failure \
    -R "^test_internals$"
ISOLATED_STATUS=$?
set -e
echo "Isolated BRGEMM diagnostic exit code: ${ISOLATED_STATUS}"
echo

# --- Step 4: Run targeted AMX bf16 tests ---------------------------------------
echo "------------------------------------------------------------------"
echo " Step 4: Run targeted AMX bf16 tests (ONEDNN_VERBOSE=1)"
echo "------------------------------------------------------------------"
export ONEDNN_VERBOSE=1
ctest --test-dir "${BUILD_DIR}" --verbose --output-on-failure \
    -R "test_internals|test_benchdnn_modeC_conv_smoke_cpu|test_benchdnn_modeC_matmul_smoke_cpu"
