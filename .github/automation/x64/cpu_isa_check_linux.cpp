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

// Standalone diagnostic tool (Linux/GCC/Clang) that reports the CPU's Intel AMX
// and AVX-512 BF16 capabilities as seen through CPUID + XCR0, interprets those
// bits the same way oneDNN's ISA detection does, additionally performs the Linux
// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) handshake that oneDNN uses
// on Linux, and finally performs a *guarded* AMX instruction probe (SIGILL via
// sigsetjmp/siglongjmp) to empirically confirm whether tile instructions
// actually execute on this host/VM or raise #UD.
//
// This never aborts: an illegal-instruction fault during the AMX probe is
// caught and reported, so it is safe to run on VMs that mis-advertise AMX.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <csetjmp>
#include <csignal>

#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

struct cpuid_regs_t {
    unsigned int eax, ebx, ecx, edx;
};

cpuid_regs_t cpuid(unsigned int leaf, unsigned int subleaf) {
    cpuid_regs_t r {0, 0, 0, 0};
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
    return r;
}

// Read an extended control register (XCR0 == index 0) via XGETBV.
uint64_t read_xcr(unsigned int index) {
    unsigned int eax = 0, edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
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
    const char *p = brand;
    while (*p == ' ')
        ++p;
    printf("CPU brand      : %s\n", p);
}

const char *microarch_name(
        unsigned int display_family, unsigned int display_model) {
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

// oneDNN's Linux AMX handshake. Mirrors amx::init() in cpu_isa_traits.cpp:
// request permission to use XTILEDATA from the kernel and confirm it is granted.
// Returns 1 = granted, 0 = denied/failed.
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#define XFEATURE_MASK_XTILEDATA (1u << XFEATURE_XTILEDATA)

int amx_arch_prctl_handshake() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (status != 0) return 0;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return 1;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (status != 0) return 0;
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (status != 0 || !(bitmask & XFEATURE_MASK_XTILEDATA)) return 0;
    return 1;
}

// Guarded AMX execution probe. Attempts to execute a real tile instruction and
// catches SIGILL (#UD) via sigsetjmp/siglongjmp.
// Returns 1 = executed OK, 0 = raised an exception (e.g. #UD / illegal instr).
sigjmp_buf g_jmp;
void sigill_handler(int) {
    siglongjmp(g_jmp, 1);
}

int probe_amx_execution() {
    struct sigaction sa {};
    struct sigaction old {};
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old);

    int result;
    if (sigsetjmp(g_jmp, 1) == 0) {
        // TILERELEASE (encoded as bytes: C4 E2 78 49 C0) is a benign AMX tile
        // instruction that resets tile state and needs no configuration. If AMX
        // is genuinely usable this is a no-op; otherwise it raises #UD.
        __asm__ __volatile__(".byte 0xC4, 0xE2, 0x78, 0x49, 0xC0" ::: "memory");
        result = 1;
    } else {
        result = 0;
    }

    sigaction(SIGILL, &old, nullptr);
    return result;
}

} // namespace

int main() {
    printf("==================================================================\n");
    printf(" oneDNN CPU ISA diagnostic (AMX / AVX-512 BF16) [Linux]\n");
    printf("==================================================================\n");

    print_brand_string();
    print_microarch();

    // --- Raw feature registers ------------------------------------------------
    const cpuid_regs_t leaf1 = cpuid(0x1u, 0);
    const cpuid_regs_t leaf7_0 = cpuid(0x7u, 0);
    const cpuid_regs_t leaf7_1 = cpuid(0x7u, 1);

    const bool osxsave = bit(leaf1.ecx, 27);
    uint64_t xcr0 = 0;
    if (osxsave) xcr0 = read_xcr(0);

    printf("\n-- Raw registers --------------------------------------------------\n");
    printf("CPUID.1:ECX    = 0x%08X\n", leaf1.ecx);
    printf("CPUID.7.0:EBX  = 0x%08X\n", leaf7_0.ebx);
    printf("CPUID.7.0:ECX  = 0x%08X\n", leaf7_0.ecx);
    printf("CPUID.7.0:EDX  = 0x%08X\n", leaf7_0.edx);
    printf("CPUID.7.1:EAX  = 0x%08X\n", leaf7_1.eax);
    printf("XCR0           = 0x%016llX\n",
            static_cast<unsigned long long>(xcr0));

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

    // --- Derived verdicts (mirror oneDNN mayiuse() logic) ---------------------
    const bool bf16_usable = avx512f && avx512dq && avx512bw && avx512vl
            && avx512_vnni && avx512_bf16;

    // oneDNN Linux amx::is_available() relies on the arch_prctl handshake rather
    // than a raw XCR0 read, so perform it here for an apples-to-apples verdict.
    const int prctl_ok = amx_arch_prctl_handshake();
    const bool amx_os_ok = osxsave && xtilecfg && xtiledata;
    const bool amx_selected = amx_tile && prctl_ok;
    const bool amx_bf16_selected = amx_selected && amx_bf16;

    printf("\n-- oneDNN ISA verdicts --------------------------------------------\n");
    printf("AVX512 BF16 usable (avx512_core_bf16) : %s\n", yn(bf16_usable));
    printf("XCR0 tile bits set (raw)              : %s\n", yn(amx_os_ok));
    printf("arch_prctl XTILEDATA granted          : %s\n", yn(prctl_ok != 0));
    printf("oneDNN would enable AMX (amx_tile)    : %s\n", yn(amx_selected));
    printf("oneDNN would enable AMX BF16          : %s\n", yn(amx_bf16_selected));

    // --- Guarded AMX execution probe ------------------------------------------
    printf("\n-- AMX execution probe (guarded) ----------------------------------\n");
    if (!amx_tile) {
        printf("AMX EXECUTION      : SKIPPED (AMX_TILE not advertised)\n");
    } else {
        const int ok = probe_amx_execution();
        if (ok) {
            printf("AMX EXECUTION      : OK (tile instruction executed)\n");
        } else {
            printf("AMX EXECUTION      : FAULTED (#UD) - AMX advertised but not "
                   "executable on this host\n");
        }
    }

    printf("==================================================================\n");
    return 0;
}
