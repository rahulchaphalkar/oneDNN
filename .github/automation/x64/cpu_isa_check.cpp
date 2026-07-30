/*******************************************************************************
* Copyright 2026 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Standalone diagnostic tool (Windows/MSVC) that reports the CPU's Intel AMX
// and AVX-512 BF16 capabilities as seen through CPUID + XCR0, interprets those
// bits the same way oneDNN's ISA detection does, and finally performs a
// *guarded* AMX instruction probe (under SEH) to empirically confirm whether
// tile instructions actually execute on this host/VM or raise #UD.
//
// This never aborts: an illegal-instruction fault during the AMX probe is
// caught and reported, so it is safe to run on VMs that mis-advertise AMX.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <intrin.h>
#include <immintrin.h>
#include <windows.h>

namespace {

struct cpuid_regs_t {
    unsigned int eax, ebx, ecx, edx;
};

cpuid_regs_t cpuid(unsigned int leaf, unsigned int subleaf) {
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    cpuid_regs_t r;
    r.eax = static_cast<unsigned int>(regs[0]);
    r.ebx = static_cast<unsigned int>(regs[1]);
    r.ecx = static_cast<unsigned int>(regs[2]);
    r.edx = static_cast<unsigned int>(regs[3]);
    return r;
}

bool bit(unsigned int value, int pos) {
    return ((value >> pos) & 1u) != 0u;
}

const char *yn(bool v) {
    return v ? "yes" : "no";
}

void print_brand_string() {
    // Extended CPUID leaves 0x80000002-0x80000004 hold the 48-byte brand string.
    cpuid_regs_t ext = cpuid(0x80000000u, 0);
    if (ext.eax < 0x80000004u) {
        printf("CPU brand      : <not available>\n");
        return;
    }
    char brand[49] = {0};
    for (unsigned int i = 0; i < 3; ++i) {
        cpuid_regs_t r = cpuid(0x80000002u + i, 0);
        std::memcpy(brand + i * 16 + 0, &r.eax, 4);
        std::memcpy(brand + i * 16 + 4, &r.ebx, 4);
        std::memcpy(brand + i * 16 + 8, &r.ecx, 4);
        std::memcpy(brand + i * 16 + 12, &r.edx, 4);
    }
    // Trim leading spaces for readability.
    const char *p = brand;
    while (*p == ' ')
        ++p;
    printf("CPU brand      : %s\n", p);
}

const char *microarch_name(unsigned int display_family, unsigned int display_model) {
    if (display_family != 0x6) return "unknown (non family-6)";
    switch (display_model) {
        case 0x8F: return "Sapphire Rapids (SPR)";
        case 0xCF: return "Emerald Rapids (EMR)";
        case 0xAD: return "Granite Rapids (GNR)";
        case 0xAE: return "Granite Rapids-D (GNR-D)";
        case 0x6A: return "Ice Lake-SP (ICX)";
        case 0x6C: return "Ice Lake-D";
        default: return "unknown model";
    }
}

void print_microarch() {
    cpuid_regs_t r = cpuid(0x1u, 0);
    const unsigned int eax = r.eax;
    unsigned int base_family = (eax >> 8) & 0xF;
    unsigned int base_model = (eax >> 4) & 0xF;
    unsigned int ext_family = (eax >> 20) & 0xFF;
    unsigned int ext_model = (eax >> 16) & 0xF;

    unsigned int display_family = base_family;
    if (base_family == 0xF) display_family += ext_family;
    unsigned int display_model = base_model;
    if (base_family == 0x6 || base_family == 0xF)
        display_model += (ext_model << 4);

    printf("Family/Model   : family=0x%X model=0x%X -> %s\n", display_family,
            display_model, microarch_name(display_family, display_model));
}

// AMX tile configuration structure (palette 1): 64-byte layout expected by
// LDTILECFG. Mirrors the layout oneDNN builds in brgemm_init_tiles.
struct amx_tilecfg_t {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

// Result of the staged AMX execution probe: identifies exactly how far the
// tile pipeline got before faulting (if at all).
enum class amx_probe_stage_t {
    ok, // full-size and small-tail BF16 probes executed
    release_faulted, // even TILERELEASE (no config/data) raised #UD
    ldtilecfg_faulted, // tile *configuration* raised #UD
    tileload_dpbf16_faulted, // tile *data*/compute raised #UD
    small_bf16_faulted, // valid M=N=K=4 BF16 tile geometry raised #UD
};

// Level 1 probe: TILERELEASE only. Needs no tile config or data - the cheapest
// possible tile instruction. Passing this does NOT prove the data/compute path
// works (some VMs enable config-level ops but not XTILEDATA-backed ops).
bool probe_tilerelease() {
    __try {
        _tile_release();
        return true;
    } __except (GetExceptionCode() == static_cast<DWORD>(STATUS_ILLEGAL_INSTRUCTION)
                    ? EXCEPTION_EXECUTE_HANDLER
                    : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// Level 2 probe: LDTILECFG with a valid palette-1 configuration. Exercises the
// tile *configuration* path but not tile data loads or the multiply.
bool probe_ldtilecfg(const amx_tilecfg_t *cfg) {
    __try {
        _tile_loadconfig(cfg);
        _tile_release();
        return true;
    } __except (GetExceptionCode() == static_cast<DWORD>(STATUS_ILLEGAL_INSTRUCTION)
                    ? EXCEPTION_EXECUTE_HANDLER
                    : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// Level 3 probe: full path - LDTILECFG + TILELOADD (loads tile *data*, needs
// XTILEDATA to be genuinely backed by the OS/hypervisor) + TDPBF16PS (the
// bf16 tile multiply, exactly what oneDNN's matmul/conv kernels run). This is
// the sequence that faults on VMs with only partial AMX enablement.
bool probe_tdpbf16ps(const amx_tilecfg_t *cfg, const uint16_t *a,
        const uint16_t *b, float *c, int stride) {
    __try {
        _tile_loadconfig(cfg);
        _tile_loadd(0, a, stride); // tmm0 = A
        _tile_loadd(1, b, stride); // tmm1 = B
        _tile_loadd(2, c, stride); // tmm2 = C (accumulator)
        _tile_dpbf16ps(2, 0, 1); // tmm2 += tmm0 * tmm1  (bf16)
        _tile_stored(2, c, stride);
        _tile_release();
        return true;
    } __except (GetExceptionCode() == static_cast<DWORD>(STATUS_ILLEGAL_INSTRUCTION)
                    ? EXCEPTION_EXECUTE_HANDLER
                    : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// Staged AMX execution probe. Runs progressively heavier tile operations and
// reports the first stage that faults, so a VM that enables config-level tile
// ops but not XTILEDATA-backed data/compute ops (the windows-2025 case) is
// distinguished from one where AMX is entirely unusable.
amx_probe_stage_t probe_amx_execution() {
    if (!probe_tilerelease()) return amx_probe_stage_t::release_faulted;

    // Consistent palette-1 config: M=16, N=16, K=32 bf16.
    // A: rows=16, colsb=64 (32 bf16 = K).  B: rows=16, colsb=64 (pre-packed,
    // N*4).  C: rows=16, colsb=64 (N*4 f32).  All tiles uniformly 16x64.
    amx_tilecfg_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    for (int t = 0; t < 3; ++t) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }

    if (!probe_ldtilecfg(&cfg)) return amx_probe_stage_t::ldtilecfg_faulted;

    // Tile data buffers. 16 rows * 64 bytes = 1024 bytes each.
    alignas(64) uint16_t a[16 * 32];
    alignas(64) uint16_t b[16 * 32];
    alignas(64) float c[16 * 16];
    for (int i = 0; i < 16 * 32; ++i) {
        a[i] = 0x3F80; // bf16(1.0)
        b[i] = 0x3F80; // bf16(1.0)
    }
    std::memset(c, 0, sizeof(c));

    if (!probe_tdpbf16ps(&cfg, a, b, c, 64))
        return amx_probe_stage_t::tileload_dpbf16_faulted;

    // Repeat the BF16 multiply with the exact small geometry that starts the
    // failing oneDNN BRGEMM case /54: M=4, N=4, K=4.
    // A: rows=M=4, colsb=K*2=8.
    // B: rows=K/2=2, colsb=N*4=16 (VNNI-packed BF16 pairs).
    // C: rows=M=4, colsb=N*4=16.
    amx_tilecfg_t small_cfg;
    std::memset(&small_cfg, 0, sizeof(small_cfg));
    small_cfg.palette_id = 1;
    small_cfg.rows[0] = 4;
    small_cfg.colsb[0] = 8;
    small_cfg.rows[1] = 2;
    small_cfg.colsb[1] = 16;
    small_cfg.rows[2] = 4;
    small_cfg.colsb[2] = 16;
    if (!probe_tdpbf16ps(&small_cfg, a, b, c, 64))
        return amx_probe_stage_t::small_bf16_faulted;

    return amx_probe_stage_t::ok;
}

} // namespace

int main() {
    printf("==================================================================\n");
    printf(" oneDNN CPU ISA diagnostic (AMX / AVX-512 BF16)\n");
    printf("==================================================================\n");

    print_brand_string();
    print_microarch();

    // --- Raw feature registers ------------------------------------------------
    const cpuid_regs_t leaf1 = cpuid(0x1u, 0);
    const cpuid_regs_t leaf7_0 = cpuid(0x7u, 0);
    const cpuid_regs_t leaf7_1 = cpuid(0x7u, 1);
    const cpuid_regs_t leaf1d_0 = cpuid(0x1Du, 0);
    const cpuid_regs_t leaf1d_1 = cpuid(0x1Du, 1);

    // OSXSAVE (CPUID.1:ECX[27]) tells us XGETBV is usable.
    const bool osxsave = bit(leaf1.ecx, 27);
    uint64_t xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);

    printf("\n-- Raw registers --------------------------------------------------\n");
    printf("CPUID.1:ECX    = 0x%08X\n", leaf1.ecx);
    printf("CPUID.7.0:EBX  = 0x%08X\n", leaf7_0.ebx);
    printf("CPUID.7.0:ECX  = 0x%08X\n", leaf7_0.ecx);
    printf("CPUID.7.0:EDX  = 0x%08X\n", leaf7_0.edx);
    printf("CPUID.7.1:EAX  = 0x%08X\n", leaf7_1.eax);
        printf("CPUID.1D.0:EAX = 0x%08X (max AMX palette ID)\n", leaf1d_0.eax);
        printf("CPUID.1D.1:EBX = 0x%08X (palette 1 tiles/colsb)\n",
            leaf1d_1.ebx);
        printf("CPUID.1D.1:ECX = 0x%08X (palette 1 rows)\n", leaf1d_1.ecx);
    printf("XCR0           = 0x%016llX\n", static_cast<unsigned long long>(xcr0));

    // --- Parsed feature bits --------------------------------------------------
    const bool avx512f = bit(leaf7_0.ebx, 16);
    const bool avx512dq = bit(leaf7_0.ebx, 17);
    const bool avx512bw = bit(leaf7_0.ebx, 30);
    const bool avx512vl = bit(leaf7_0.ebx, 31);
    const bool avx512_vnni = bit(leaf7_0.ecx, 11);
    const bool avx512_bf16 = bit(leaf7_1.eax, 5);
    const bool avx512_fp16 = bit(leaf7_0.edx, 23);
    const bool amx_bf16 = bit(leaf7_0.edx, 22);
    const bool amx_tile = bit(leaf7_0.edx, 24);
    const bool amx_int8 = bit(leaf7_0.edx, 25);
    const bool xtilecfg = bit(static_cast<unsigned int>(xcr0), 17);
    const bool xtiledata = bit(static_cast<unsigned int>(xcr0), 18);
    const unsigned int max_palette = leaf1d_0.eax;
    const unsigned int palette1_max_tiles = leaf1d_1.ebx >> 16;
    const unsigned int palette1_max_colsb = leaf1d_1.ebx & 0xFFFFu;
    const unsigned int palette1_max_rows = leaf1d_1.ecx & 0xFFFFu;

    printf("\n-- Parsed capabilities --------------------------------------------\n");
    printf("OSXSAVE            : %s\n", yn(osxsave));
    printf("AVX512F            : %s\n", yn(avx512f));
    printf("AVX512DQ           : %s\n", yn(avx512dq));
    printf("AVX512BW           : %s\n", yn(avx512bw));
    printf("AVX512VL           : %s\n", yn(avx512vl));
    printf("AVX512_VNNI        : %s\n", yn(avx512_vnni));
    printf("AVX512_BF16        : %s\n", yn(avx512_bf16));
    printf("AVX512_FP16        : %s\n", yn(avx512_fp16));
    printf("AMX_TILE           : %s\n", yn(amx_tile));
    printf("AMX_BF16           : %s\n", yn(amx_bf16));
    printf("AMX_INT8           : %s\n", yn(amx_int8));
    printf("XCR0.XTILECFG (17) : %s\n", yn(xtilecfg));
    printf("XCR0.XTILEDATA(18) : %s\n", yn(xtiledata));
        printf("AMX max palette ID : %u\n", max_palette);
        printf("AMX palette 1      : tiles=%u colsb=%u rows=%u\n",
            palette1_max_tiles, palette1_max_colsb, palette1_max_rows);

    // --- Derived verdicts (mirror oneDNN mayiuse() logic) ---------------------
    const bool bf16_usable = avx512f && avx512dq && avx512bw && avx512vl
            && avx512_vnni && avx512_bf16;
    // oneDNN Windows amx::is_available(): OSXSAVE && XCR0[17] && XCR0[18].
    const bool amx_os_ok = osxsave && xtilecfg && xtiledata;
    const bool amx_selected = amx_tile && amx_os_ok;
    const bool amx_bf16_selected = amx_selected && amx_bf16;

    printf("\n-- oneDNN ISA verdicts --------------------------------------------\n");
    printf("AVX512 BF16 usable (avx512_core_bf16) : %s\n", yn(bf16_usable));
    printf("oneDNN would enable AMX (amx_tile)    : %s\n", yn(amx_selected));
    printf("oneDNN would enable AMX BF16          : %s\n", yn(amx_bf16_selected));
    printf("oneDNN AMX target palette             : %u\n",
            max_palette >= 1 ? 1u : 0u);
    if (amx_selected && max_palette == 0) {
        printf("WARNING: inconsistent AMX enumeration: AMX_TILE + XCR0 are "
               "enabled, but CPUID.1D reports no palette. oneDNN will load "
               "palette 0 (INIT state), so subsequent tile data/compute "
               "instructions can raise #UD.\n");
    }

    // --- Guarded AMX execution probe ------------------------------------------
    printf("\n-- AMX execution probe (guarded, staged) --------------------------\n");
    if (!amx_selected) {
        printf("AMX EXECUTION      : SKIPPED (AMX not advertised/enabled)\n");
    } else {
        const amx_probe_stage_t stage = probe_amx_execution();
        switch (stage) {
            case amx_probe_stage_t::ok:
                  printf("AMX EXECUTION      : OK (full M=16,N=16,K=32 and "
                      "small M=N=K=4 BF16 tile paths executed)\n");
                break;
            case amx_probe_stage_t::release_faulted:
                printf("AMX EXECUTION      : FAULTED (#UD) at TILERELEASE - no "
                       "tile ops usable on this host\n");
                break;
            case amx_probe_stage_t::ldtilecfg_faulted:
                printf("AMX EXECUTION      : FAULTED (#UD) at LDTILECFG - tile "
                       "configuration not usable on this host\n");
                break;
            case amx_probe_stage_t::tileload_dpbf16_faulted:
                printf("AMX EXECUTION      : FAULTED (#UD) at TILELOADD/TDPBF16PS "
                       "- config OK but XTILEDATA-backed data/compute NOT usable "
                       "(partial AMX enablement, e.g. under virtualization)\n");
                break;
                 case amx_probe_stage_t::small_bf16_faulted:
                  printf("AMX EXECUTION      : FAULTED (#UD) in valid small "
                      "M=N=K=4 TDPBF16PS path (full-tile path passed)\n");
                  break;
        }
    }

    printf("==================================================================\n");
    return 0;
}
