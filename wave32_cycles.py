#!/usr/bin/env python3
"""
wave32_cycles.py — Cycle-Exact GFX11 (RDNA3) Wave32 Binary ISA, AMDGPU ELF64,
Real Hardware SQTT v2 Nibble-Packet & 32-Bank LDS Pipeline Cycle Emulator (DISPATCH-09 REV=3).

Zero external dependencies (Python 3.12+ stdlib only).

Capabilities:
  1. Complete little-endian 32-bit/64-bit GFX11 (RDNA3 Wave32) binary ISA decoder
     `decode_gfx11(code_bytes) -> list[DecodedInsn]` + matching encoders covering ALL 20
     RDNA3 instruction formats (cross-verified against LLVM/Clang 21 `-mcpu=gfx1100`):
       - SOPP (s_waitcnt vmcnt/lgkmcnt/expcnt, s_delay_alu instid0/instskip/instid1,
         s_clause, s_waitcnt_depctr, s_code_end, s_barrier, s_branch/s_cbranch_*, s_endpgm).
       - SOPK (s_movk_i32, s_addk_i32, s_mulk_i32, s_cmpk_*, s_call_b64, s_setreg_imm32_b32,
         s_waitcnt_vscnt [0xBC7C0000], s_waitcnt_vmcnt, s_waitcnt_expcnt, s_waitcnt_lgkmcnt).
       - SOP1, SOP2, SOPC (full scalar arithmetic/logic/compare + s_*_saveexec_b32/b64).
       - VOP2, VOP1 (incl. VALUT_4 transcendentals & f64 VALUB_16), VOPC (`v_cmp_*` -> vcc_lo),
         VOPCX (`v_cmpx_*` op>=0x80 -> exec_lo, `Pipe.VALU1_WR_EXEC`, `writes_exec=True`).
       - VOP3 / VOP3SD (incl. v_cmp*_e64, v_cmpx*_e64, VOP2/VOP1 _e64, v_lshl_or_b32,
         v_lshl_add_u32, v_add_co_u32, v_fma_f32/f64, v_lshlrev_b64, v_mul_lo_u32).
       - VOP3P (v_pk_*, v_dot2/4/8_*, v_fma_mix_*, v_wmma_* matrix ops [0xCC400000]).
       - VOPD (64-bit + optional 32b literal X/Y dual-VALU pair with opposite dst banks).
       - DS (ds_load/store_b{8,16,32,64,96,128}, 2addr ops, atomics, swizzle/permute).
       - FLAT / SCRATCH / GLOBAL (seg=0/1/2, loads/stores/atomics, 32b saddr vs 64b vaddr).
       - MUBUF (`buffer_load_b32` [0xE0500000]..b128, `buffer_store_*`, format, inv),
         MTBUF, MIMG, SMEM (`s_load_b{32..512}`, `s_buffer_load_*`), VINTERP, LDSDIR, EXP.
  2. Native AMDGPU ELF64 parser (`parse_amdgpu_elf`, `decode_elf`) supporting both full
     LLVM/lld AMDGPU ELFs (`EM_AMDGPU=224`) and minimal HSA kernel ELFs, extracting `.text`,
     `.rodata` AMD HSA kernel descriptor (LDS/scratch/kernarg/wavefront32), and symbols.
  3. Genuine AMD GFX11 (RDNA3 Layout 3 & RDNA4 Layout 4) hardware SQTT v2 nibble-packet
     decoder & encoder (`parse_sqtt`, `encode_sqtt`, `align_sqtt_to_program`,
     `extract_sqtt_contention`, `sqtt_to_tuples`) compatible with real ROCprofv3 / AMDGPU
     KFD hardware traces — zero invented packet headers or pre-baked timing fields.
  4. Real 32-lane Wave32 functional ISA execution engine (`Wave32ISAState`) integrated inside
     `Wave32Simulator` (`simulate_wave`, `simulate_cu`): executes VGPR/SGPR/SCC/VCC/EXEC/LDS
     semantics dynamically, computing exact per-lane LDS byte addresses, live EXEC masks,
     conditional branches & loops, `s_delay_alu` scheduling hints, `vscnt`/`vmcnt`/`lgkmcnt`
     waits, and automatic VALU-only bypass without requiring manual overrides.
  5. Real least-squares / coordinate-descent calibrator `fit_profile(traces)` and real
     hardware AMD Radeon RX 7900 XTX (`gfx1100`) ELF + SQTT v2 validation (`validate_hardware_sqtt`)
     on unmodified hardware captures (`E_3`, `custom_lds_sync`, `r_32_32_16_2`).

Run self-tests:
    python3 -W error wave32_cycles.py --selftest
    python3 -W error wave32_cycles.py --validate-hw
    python3 -W error -m unittest -v wave32_cycles.py
    python3 -W error -m pytest -v wave32_cycles.py
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import struct
import sys
import unittest
import zlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Architectural Enums & Constants
# ---------------------------------------------------------------------------

FULL_EXEC_MASK: int = 0xFFFFFFFF
LDS_NUM_BANKS: int = 32
LDS_BANK_SHIFT: int = 2  # 4-byte (1 dword) bank width
HALF_WAVE_LANES: int = 16
WAVE_LANES: int = 32
CU_NUM_SIMDS: int = 4
EXEC_REG: str = "exec_lo"
EXEC_LO_ENC: int = 126
EXEC_HI_ENC: int = 127
NULL_ENC: int = 124
VCC_LO_ENC: int = 106
VCC_HI_ENC: int = 107
LITERAL_ENC: int = 255
DPP16_ENC: int = 250
SDWA_ENC: int = 249
DPP8_ENC: int = 233
DPP8FI_ENC: int = 234
WAITCNT_VM_NONE: int = 63
WAITCNT_LGKM_NONE: int = 63
WAITCNT_EXP_NONE: int = 7
WAITCNT_VS_NONE: int = 63
EM_AMDGPU: int = 224


def _canon_reg(name: str) -> str:
    if name in ("exec", "exec_lo", "exec_hi"):
        return EXEC_REG
    return name


class AluSrc(str, Enum):
    """Issue-slot / ALU source classification for Wave32 front-end packets."""

    VALU = "VALU"
    SALU = "SALU"
    VALU_SALU = "VALU_SALU"
    VOPD = "VOPD"
    LDS = "LDS"
    SMEM = "SMEM"
    VMEM = "VMEM"
    BRANCH = "BRANCH"
    BARRIER = "BARRIER"
    WAITCNT = "WAITCNT"
    ENDPGM = "ENDPGM"


class Pipe(str, Enum):
    """Wave32 execution pipes & memory/synchronization operations."""

    SALU = "SALU"
    SALU_WR_EXEC = "SALU_WR_EXEC"
    JUMP = "JUMP"
    BARRIER = "BARRIER"
    VALU32 = "VALU32"
    VALU1_WR_EXEC = "VALU1_WR_EXEC"
    VALUT_4 = "VALUT_4"
    VALUB_2 = "VALUB_2"
    VALUB_4 = "VALUB_4"
    VALUB_16 = "VALUB_16"
    VALU_WMMA = "VALU_WMMA"
    VINTERP = "VINTERP"
    LDS_RD = "LDS_RD"
    LDS_WR_1 = "LDS_WR_1"
    LDS_WR_2 = "LDS_WR_2"
    LDS_WR_3 = "LDS_WR_3"
    LDS_WR_4 = "LDS_WR_4"
    LDS_WR_5 = "LDS_WR_5"
    SMEM_RD = "SMEM_RD"
    VMEM_RD = "VMEM_RD"
    VMEM_WR = "VMEM_WR"
    WAIT_LGKMCNT = "WAIT_LGKMCNT"
    DELAY_ALU = "DELAY_ALU"
    ENDPGM = "ENDPGM"


class ContentionPort(str, Enum):
    """Shared CU SRAM / memory ports subject to cross-SIMD background contention."""

    OTHER_LDS = "OTHER_LDS"
    OTHER_VMEM = "OTHER_VMEM"


VALU_PIPES: frozenset[Pipe] = frozenset({
    Pipe.VALU32,
    Pipe.VALU1_WR_EXEC,
    Pipe.VALUT_4,
    Pipe.VALUB_2,
    Pipe.VALUB_4,
    Pipe.VALUB_16,
    Pipe.VALU_WMMA,
    Pipe.VINTERP,
})

SALU_PIPES: frozenset[Pipe] = frozenset({
    Pipe.SALU,
    Pipe.SALU_WR_EXEC,
})

LDS_WR_PIPES: Dict[Pipe, int] = {
    Pipe.LDS_WR_1: 1,
    Pipe.LDS_WR_2: 2,
    Pipe.LDS_WR_3: 3,
    Pipe.LDS_WR_4: 4,
    Pipe.LDS_WR_5: 5,
}

LDS_WR_BY_DWORDS: Dict[int, Pipe] = {v: k for k, v in LDS_WR_PIPES.items()}
LDS_PIPES: frozenset[Pipe] = frozenset({Pipe.LDS_RD, *LDS_WR_PIPES.keys()})
LGKM_PIPES: frozenset[Pipe] = frozenset({Pipe.LDS_RD, *LDS_WR_PIPES.keys(), Pipe.SMEM_RD})
VMEM_PIPES: frozenset[Pipe] = frozenset({Pipe.VMEM_RD, Pipe.VMEM_WR})


# ---------------------------------------------------------------------------
# LatencyProfile & Instruction / Trace Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatencyProfile:
    """Parameterizable Wave32 microarchitectural latency and issue-occupancy table."""

    salu_issue: int = 1
    salu_latency: int = 1
    salu_wr_exec_issue: int = 2
    salu_wr_exec_latency: int = 2
    jump_taken_cycles: int = 4
    jump_untaken_cycles: int = 1
    barrier_cycles: int = 1
    waitcnt_cycles: int = 1

    valu32_issue: int = 1
    valu32_latency: int = 4
    valu32_forwarding: bool = False
    valu32_forward_latency: int = 1
    valu_wr_exec_issue: int = 1
    valu_wr_exec_latency: int = 4
    valut_4_issue: int = 4
    valut_4_latency: int = 16
    valub_2_issue: int = 2
    valub_2_latency: int = 2
    valub_4_issue: int = 4
    valub_4_latency: int = 4
    valub_16_issue: int = 16
    valub_16_latency: int = 16
    valu_wmma_issue: int = 16
    valu_wmma_latency: int = 32
    vinterp_issue: int = 2
    vinterp_latency: int = 8

    lds_rd_base: int = 2
    lds_rd_latency: int = 20
    lds_wr_latency_overhead: int = 8
    smem_rd_issue: int = 1
    smem_rd_latency: int = 20
    vmem_rd_issue: int = 4
    vmem_rd_latency: int = 120
    vmem_wr_issue: int = 4
    vmem_wr_latency: int = 40
    # Optional cross-domain VALU -> LDS/VMEM address/data issue latency (0 = standard
    # synthetic profile; ~17 cycles on real GFX1100 RX 7900 XTX silicon).
    valu_to_mem_latency: int = 0
    # Optional SALU -> VMEM/LDS issue latency (0 = standard synthetic profile).
    salu_to_mem_latency: int = 0
    # Optional post-waitcnt front-end resume gap (0 = immediate on synthetic profile).
    waitcnt_resume_gap: int = 0

    enable_dual_issue: bool = True

    def pipe_timing(self, pipe: Pipe, taken: bool = False) -> Tuple[int, int]:
        """Return (base_issue_occupancy, base_result_latency) for non-LDS pipes."""
        if pipe is Pipe.SALU:
            return self.salu_issue, self.salu_latency
        if pipe is Pipe.SALU_WR_EXEC:
            return self.salu_wr_exec_issue, self.salu_wr_exec_latency
        if pipe is Pipe.JUMP:
            cyc = self.jump_taken_cycles if taken else self.jump_untaken_cycles
            return cyc, cyc
        if pipe is Pipe.BARRIER:
            return self.barrier_cycles, self.barrier_cycles
        if pipe is Pipe.WAIT_LGKMCNT:
            return self.waitcnt_cycles, self.waitcnt_cycles
        if pipe is Pipe.DELAY_ALU:
            return 0, 0
        if pipe is Pipe.ENDPGM:
            return 1, 1
        if pipe is Pipe.VALU32:
            return self.valu32_issue, self.valu32_latency
        if pipe is Pipe.VALU1_WR_EXEC:
            return self.valu_wr_exec_issue, self.valu_wr_exec_latency
        if pipe is Pipe.VALUT_4:
            return self.valut_4_issue, self.valut_4_latency
        if pipe is Pipe.VALUB_2:
            return self.valub_2_issue, self.valub_2_latency
        if pipe is Pipe.VALUB_4:
            return self.valub_4_issue, self.valub_4_latency
        if pipe is Pipe.VALUB_16:
            return self.valub_16_issue, self.valub_16_latency
        if pipe is Pipe.VALU_WMMA:
            return self.valu_wmma_issue, self.valu_wmma_latency
        if pipe is Pipe.VINTERP:
            return self.vinterp_issue, self.vinterp_latency
        if pipe is Pipe.SMEM_RD:
            return self.smem_rd_issue, self.smem_rd_latency
        if pipe is Pipe.VMEM_RD:
            return self.vmem_rd_issue, self.vmem_rd_latency
        if pipe is Pipe.VMEM_WR:
            return self.vmem_wr_issue, self.vmem_wr_latency
        raise ValueError(f"pipe_timing called for LDS pipe {pipe}; use analyze_lds_access")

    def valu32_raw_latency(self, insn_forwarded: bool, consumer_eligible: bool = True) -> int:
        """RAW latency from a VALU32 producer.

        Bypass forwarding ONLY applies to eligible VALU consumers (never across
        into SALU, DS/LDS, VMEM, SMEM, or JUMP address/condition operands).
        """
        if consumer_eligible and (self.valu32_forwarding or insn_forwarded):
            return min(self.valu32_forward_latency, self.valu32_latency)
        return self.valu32_latency


@dataclass(frozen=True)
class Instruction:
    """Single Wave32 instruction for pipeline, ISA, and LDS cycle simulation."""

    pc: int
    pipe: Pipe
    dst: Tuple[str, ...] = ()
    src: Tuple[str, ...] = ()
    taken: bool = False
    forwarded: bool = False
    lgkm_target: int = WAITCNT_LGKM_NONE
    vm_target: int = WAITCNT_VM_NONE
    exp_target: int = WAITCNT_EXP_NONE
    vs_target: int = WAITCNT_VS_NONE
    barrier_id: int = 0
    exec_mask: Optional[int] = None
    lds_addrs: Optional[Tuple[int, ...]] = None
    lds_dwords: int = 1
    lds_sub_offsets: Optional[Tuple[int, ...]] = None
    is_vopd: bool = False
    mnemonic: str = ""
    decoded: Optional["DecodedInsn"] = None

    def effective_dst(self) -> Tuple[str, ...]:
        regs = [_canon_reg(r) for r in self.dst]
        if self.pipe in (Pipe.SALU_WR_EXEC, Pipe.VALU1_WR_EXEC) and EXEC_REG not in regs:
            regs.append(EXEC_REG)
        return tuple(regs)

    def effective_src(self) -> Tuple[str, ...]:
        regs = [_canon_reg(r) for r in self.src]
        if (self.pipe in VALU_PIPES or self.pipe in LDS_PIPES or self.pipe in VMEM_PIPES) and EXEC_REG not in regs:
            regs.append(EXEC_REG)
        return tuple(regs)


@dataclass(frozen=True)
class ContentionEvent:
    """Background shared-port occupancy injected by another SIMD on the same CU."""

    port: ContentionPort
    start_cycle: int
    duration_cycles: int

    def __post_init__(self) -> None:
        if self.duration_cycles <= 0:
            raise ValueError(f"duration_cycles must be > 0, got {self.duration_cycles}")
        if self.start_cycle < 0:
            raise ValueError(f"start_cycle must be >= 0, got {self.start_cycle}")


@dataclass(frozen=True)
class CycleEvent:
    """Cycle-exact execution record for one issued Wave32 instruction."""

    pc: int
    issue_cycle: int
    stall_cycles: int
    duration_cycles: int
    wave_id: int = 0
    simd_id: int = 0
    alu_src: AluSrc = AluSrc.VALU
    pipe: Pipe = Pipe.VALU32
    dual_issued: bool = False
    bank_conflicts: int = 0
    lds_port_cycles: int = 0
    port_contention_cycles: int = 0
    lgkm_inflight: int = 0
    vm_inflight: int = 0
    completion_cycle: int = 0
    exec_mask: int = FULL_EXEC_MASK
    taken: bool = False
    mnemonic: str = ""

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return canonical (pc, issue_cycle, stall_cycles, duration_cycles)."""
        return (self.pc, self.issue_cycle, self.stall_cycles, self.duration_cycles)

    def to_sqtt(self) -> Tuple[int, int, int, int]:
        """Return SQTT-compatible (pc, stall, dur, time) tuple."""
        return (self.pc, self.stall_cycles, self.duration_cycles, self.issue_cycle)


@dataclass(frozen=True)
class LDSBankAnalysis:
    """Detailed 32-bank LDS conflict breakdown across both 16-lane half-waves."""

    half0_max_way: int
    half1_max_way: int
    conflict_extra: int
    conflict_factor: int
    port_cycles: int
    distinct_dwords_per_bank_half0: Tuple[int, ...]
    distinct_dwords_per_bank_half1: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Complete GFX11 (RDNA3 Wave32) Binary ISA Decoder & Encoders
# ---------------------------------------------------------------------------

_SOP2_GFX11: Dict[int, str] = {
    0x00: "s_add_u32",
    0x01: "s_sub_u32",
    0x02: "s_add_i32",
    0x03: "s_sub_i32",
    0x04: "s_addc_u32",
    0x05: "s_subb_u32",
    0x06: "s_absdiff_i32",
    0x08: "s_lshl_b32",
    0x09: "s_lshl_b64",
    0x0A: "s_lshr_b32",
    0x0B: "s_lshr_b64",
    0x0C: "s_ashr_i32",
    0x0D: "s_ashr_i64",
    0x0E: "s_lshl1_add_u32",
    0x0F: "s_lshl2_add_u32",
    0x10: "s_lshl3_add_u32",
    0x11: "s_lshl4_add_u32",
    0x12: "s_min_i32",
    0x13: "s_min_u32",
    0x14: "s_max_i32",
    0x15: "s_max_u32",
    0x16: "s_and_b32",
    0x17: "s_and_b64",
    0x18: "s_or_b32",
    0x19: "s_or_b64",
    0x1A: "s_xor_b32",
    0x1B: "s_xor_b64",
    0x1C: "s_and_not1_b32",
    0x1D: "s_and_not1_b64",
    0x1E: "s_or_not1_b32",
    0x1F: "s_or_not1_b64",
    0x20: "s_nand_b32",
    0x21: "s_nand_b64",
    0x22: "s_nor_b32",
    0x23: "s_nor_b64",
    0x24: "s_xnor_b32",
    0x25: "s_xnor_b64",
    0x28: "s_bfe_u32",
    0x29: "s_bfe_i32",
    0x2A: "s_bfe_u64",
    0x2B: "s_bfe_i64",
    0x2C: "s_mul_i32",
    0x2D: "s_mul_hi_u32",
    0x2E: "s_mul_hi_i32",
    0x30: "s_cselect_b32",
    0x31: "s_cselect_b64",
    0x38: "s_pack_ll_b32_b16",
    0x39: "s_pack_lh_b32_b16",
    0x3A: "s_pack_hh_b32_b16",
}

_SOP1_GFX11: Dict[int, str] = {
    0x00: "s_mov_b32",
    0x01: "s_mov_b64",
    0x02: "s_cmov_b32",
    0x03: "s_cmov_b64",
    0x04: "s_brev_b32",
    0x05: "s_brev_b64",
    0x08: "s_ctz_i32_b32",
    0x09: "s_ctz_i32_b64",
    0x0A: "s_clz_i32_u32",
    0x0B: "s_clz_i32_u64",
    0x0C: "s_cls_i32",
    0x0D: "s_cls_i32_i64",
    0x0E: "s_sext_i32_i8",
    0x0F: "s_sext_i32_i16",
    0x10: "s_bitset0_b32",
    0x11: "s_bitset0_b64",
    0x12: "s_bitset1_b32",
    0x13: "s_bitset1_b64",
    0x14: "s_bitreplicate_b64_b32",
    0x15: "s_abs_i32",
    0x16: "s_bcnt0_i32_b32",
    0x17: "s_bcnt0_i32_b64",
    0x18: "s_bcnt1_i32_b32",
    0x19: "s_bcnt1_i32_b64",
    0x1B: "s_quadmask_b32",
    0x1C: "s_quadmask_b64",
    0x1D: "s_wqm_b32",
    0x1E: "s_not_b32",
    0x1F: "s_not_b64",
    0x20: "s_and_saveexec_b32",
    0x21: "s_and_saveexec_b64",
    0x22: "s_or_saveexec_b32",
    0x23: "s_or_saveexec_b64",
    0x24: "s_xor_saveexec_b32",
    0x25: "s_xor_saveexec_b64",
    0x26: "s_and_not1_saveexec_b32",
    0x27: "s_and_not1_saveexec_b64",
    0x28: "s_or_not1_saveexec_b32",
    0x29: "s_or_not1_saveexec_b64",
    0x2A: "s_nand_saveexec_b32",
    0x2B: "s_nand_saveexec_b64",
    0x2C: "s_nor_saveexec_b32",
    0x2D: "s_nor_saveexec_b64",
    0x2E: "s_xnor_saveexec_b32",
    0x2F: "s_xnor_saveexec_b64",
    0x30: "s_and_not0_saveexec_b32",
    0x31: "s_and_not0_saveexec_b64",
    0x32: "s_or_not0_saveexec_b32",
    0x33: "s_or_not0_saveexec_b64",
    0x4B: "s_getpc_b64",
    0x4C: "s_setpc_b64",
    0x4D: "s_swappc_b64",
}
_SOP1_SAVEEXEC_OPS: frozenset[int] = frozenset(range(0x20, 0x34))

_SOPK_GFX11: Dict[int, str] = {
    0x00: "s_movk_i32",
    0x01: "s_version",
    0x02: "s_cmovk_i32",
    0x03: "s_cmpk_eq_i32",
    0x04: "s_cmpk_lg_i32",
    0x05: "s_cmpk_gt_i32",
    0x06: "s_cmpk_ge_i32",
    0x07: "s_cmpk_lt_i32",
    0x08: "s_cmpk_le_i32",
    0x09: "s_cmpk_eq_u32",
    0x0A: "s_cmpk_lg_u32",
    0x0B: "s_cmpk_gt_u32",
    0x0C: "s_cmpk_ge_u32",
    0x0D: "s_cmpk_lt_u32",
    0x0E: "s_cmpk_le_u32",
    0x0F: "s_addk_i32",
    0x10: "s_mulk_i32",
    0x11: "s_getreg_b32",
    0x12: "s_setreg_b32",
    0x13: "s_setreg_imm32_b32",
    0x14: "s_call_b64",
    0x16: "s_subvector_loop_begin",
    0x17: "s_subvector_loop_end",
    0x18: "s_waitcnt_vscnt",
    0x19: "s_waitcnt_vmcnt",
    0x1A: "s_waitcnt_expcnt",
    0x1B: "s_waitcnt_lgkmcnt",
}
_SOPK_CMP_OPS: frozenset[int] = frozenset(range(0x03, 0x0F))
_SOPK_WAITCNT_OPS: frozenset[int] = frozenset({0x18, 0x19, 0x1A, 0x1B})

_SOPC_GFX11: Dict[int, str] = {
    0x00: "s_cmp_eq_i32",
    0x01: "s_cmp_lg_i32",
    0x02: "s_cmp_gt_i32",
    0x03: "s_cmp_ge_i32",
    0x04: "s_cmp_lt_i32",
    0x05: "s_cmp_le_i32",
    0x06: "s_cmp_eq_u32",
    0x07: "s_cmp_lg_u32",
    0x08: "s_cmp_gt_u32",
    0x09: "s_cmp_ge_u32",
    0x0A: "s_cmp_lt_u32",
    0x0B: "s_cmp_le_u32",
    0x0C: "s_bitcmp0_b32",
    0x0D: "s_bitcmp1_b32",
    0x0E: "s_bitcmp0_b64",
    0x0F: "s_bitcmp1_b64",
    0x10: "s_cmp_eq_u64",
    0x11: "s_cmp_lg_u64",
}

_SOPP_GFX11: Dict[int, str] = {
    0x00: "s_nop",
    0x01: "s_setkill",
    0x02: "s_sethalt",
    0x03: "s_sleep",
    0x04: "s_set_inst_prefetch_distance",
    0x05: "s_clause",
    0x07: "s_delay_alu",
    0x08: "s_waitcnt_depctr",
    0x09: "s_waitcnt",
    0x0A: "s_wait_idle",
    0x0B: "s_wait_event",
    0x10: "s_trap",
    0x11: "s_round_mode",
    0x12: "s_denorm_mode",
    0x1F: "s_code_end",
    0x20: "s_branch",
    0x21: "s_cbranch_scc0",
    0x22: "s_cbranch_scc1",
    0x23: "s_cbranch_vccz",
    0x24: "s_cbranch_vccnz",
    0x25: "s_cbranch_execz",
    0x26: "s_cbranch_execnz",
    0x27: "s_cbranch_cdbgsys",
    0x28: "s_cbranch_cdbguser",
    0x29: "s_cbranch_cdbgsys_or_user",
    0x2A: "s_cbranch_cdbgsys_and_user",
    0x30: "s_endpgm",
    0x31: "s_endpgm_saved",
    0x32: "s_endpgm_ordered_ps_done",
    0x34: "s_wakeup",
    0x35: "s_setprio",
    0x36: "s_sendmsg",
    0x37: "s_sendmsghalt",
    0x38: "s_incperflevel",
    0x39: "s_decperflevel",
    0x3A: "s_ttracedata",
    0x3B: "s_ttracedata_imm",
    0x3C: "s_icache_inv",
    0x3D: "s_barrier",
}
_SOPP_BRANCH_OPS: frozenset[int] = frozenset({0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A})

_VOP2_GFX11: Dict[int, str] = {
    0x01: "v_cndmask_b32",
    0x02: "v_dot2acc_f32_f16",
    0x03: "v_add_f32",
    0x04: "v_sub_f32",
    0x05: "v_subrev_f32",
    0x06: "v_fmac_dx9_zero_f32",
    0x07: "v_mul_dx9_zero_f32",
    0x08: "v_mul_f32",
    0x09: "v_mul_i32_i24",
    0x0A: "v_mul_hi_i32_i24",
    0x0B: "v_mul_u32_u24",
    0x0C: "v_mul_hi_u32_u24",
    0x0F: "v_min_f32",
    0x10: "v_max_f32",
    0x11: "v_min_i32",
    0x12: "v_max_i32",
    0x13: "v_min_u32",
    0x14: "v_max_u32",
    0x18: "v_lshlrev_b32",
    0x19: "v_lshrrev_b32",
    0x1A: "v_ashrrev_i32",
    0x1B: "v_and_b32",
    0x1C: "v_or_b32",
    0x1D: "v_xor_b32",
    0x1E: "v_xnor_b32",
    0x20: "v_add_co_ci_u32",
    0x21: "v_sub_co_ci_u32",
    0x22: "v_subrev_co_ci_u32",
    0x25: "v_add_nc_u32",
    0x26: "v_sub_nc_u32",
    0x27: "v_subrev_nc_u32",
    0x2B: "v_fmac_f32",
    0x2C: "v_fmamk_f32",
    0x2D: "v_fmaak_f32",
    0x32: "v_add_f16",
    0x33: "v_sub_f16",
    0x34: "v_subrev_f16",
    0x35: "v_mul_f16",
    0x36: "v_fmac_f16",
    0x37: "v_fmamk_f16",
    0x38: "v_fmaak_f16",
    0x39: "v_max_f16",
    0x3A: "v_min_f16",
    0x3B: "v_ldexp_f16",
}
_VOP2_LITERAL_OPS: frozenset[int] = frozenset({0x2C, 0x2D, 0x37, 0x38})

_VOP1_GFX11: Dict[int, str] = {
    0x00: "v_nop",
    0x01: "v_mov_b32",
    0x02: "v_readfirstlane_b32",
    0x03: "v_cvt_i32_f64",
    0x04: "v_cvt_f64_i32",
    0x05: "v_cvt_f32_i32",
    0x06: "v_cvt_f32_u32",
    0x07: "v_cvt_u32_f32",
    0x08: "v_cvt_i32_f32",
    0x0A: "v_cvt_f16_f32",
    0x0B: "v_cvt_f32_f16",
    0x0C: "v_cvt_nearest_i32_f32",
    0x0D: "v_cvt_floor_i32_f32",
    0x0F: "v_cvt_f32_f64",
    0x10: "v_cvt_f64_f32",
    0x15: "v_cvt_u32_f64",
    0x16: "v_cvt_f64_u32",
    0x17: "v_trunc_f64",
    0x18: "v_ceil_f64",
    0x19: "v_rndne_f64",
    0x1A: "v_floor_f64",
    0x1C: "v_mov_b16",
    0x20: "v_fract_f32",
    0x21: "v_trunc_f32",
    0x22: "v_ceil_f32",
    0x23: "v_rndne_f32",
    0x24: "v_floor_f32",
    0x25: "v_exp_f32",
    0x27: "v_log_f32",
    0x2A: "v_rcp_f32",
    0x2B: "v_rcp_iflag_f32",
    0x2E: "v_rsq_f32",
    0x2F: "v_rcp_f64",
    0x31: "v_rsq_f64",
    0x33: "v_sqrt_f32",
    0x34: "v_sqrt_f64",
    0x35: "v_sin_f32",
    0x36: "v_cos_f32",
    0x37: "v_not_b32",
    0x38: "v_bfrev_b32",
    0x3C: "v_frexp_exp_i32_f64",
    0x3D: "v_frexp_mant_f64",
    0x3E: "v_fract_f64",
    0x3F: "v_frexp_exp_i32_f32",
    0x40: "v_frexp_mant_f32",
    0x54: "v_rcp_f16",
    0x55: "v_sqrt_f16",
    0x56: "v_rsq_f16",
    0x57: "v_log_f16",
    0x58: "v_exp_f16",
    0x60: "v_sin_f16",
    0x61: "v_cos_f16",
}
_VOP1_VALUT_OPS: frozenset[int] = frozenset({
    0x25, 0x27, 0x2A, 0x2B, 0x2E, 0x33, 0x35, 0x36,
    0x54, 0x55, 0x56, 0x57, 0x58, 0x60, 0x61,
})
_VOP1_F64_OPS: frozenset[int] = frozenset({
    0x03, 0x04, 0x0F, 0x10, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
    0x2F, 0x31, 0x34, 0x3C, 0x3D, 0x3E,
})


def _build_vopc_table() -> Dict[int, str]:
    """Build complete GFX11 (RDNA3) VOPC (0x00..0x7F) & VOPCX (0x80..0xFF) opcode table."""
    tbl: Dict[int, str] = {}
    f_conds = (
        "f", "lt", "eq", "le", "gt", "lg", "ge", "o",
        "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "t",
    )
    i_conds = ("f", "lt", "eq", "le", "gt", "ne", "ge", "t")
    for is_x, base_off in ((False, 0x00), (True, 0x80)):
        pfx = "v_cmpx" if is_x else "v_cmp"
        for tbase, tname in ((0x00, "f16"), (0x10, "f32"), (0x20, "f64")):
            for idx, cond in enumerate(f_conds):
                tbl[base_off + tbase + idx] = f"{pfx}_{cond}_{tname}_e32"
        for tbase, tname in ((0x30, "i16"), (0x38, "u16"), (0x40, "i32"), (0x48, "u32"), (0x50, "i64"), (0x58, "u64")):
            for idx, cond in enumerate(i_conds):
                tbl[base_off + tbase + idx] = f"{pfx}_{cond}_{tname}_e32"
        tbl[base_off + 0x7D] = f"{pfx}_class_f16_e32"
        tbl[base_off + 0x7E] = f"{pfx}_class_f32_e32"
        tbl[base_off + 0x7F] = f"{pfx}_class_f64_e32"
    return tbl


_VOPC_GFX11: Dict[int, str] = _build_vopc_table()

# VOP3 opcode -> (mnemonic, Pipe, dst_dwords, (src0_dw, src1_dw, src2_dw), writes_sdst)
_VOP3_GFX11: Dict[int, Tuple[str, Pipe, int, Tuple[int, int, int], bool]] = {
    0x120: ("v_add_co_ci_u32", Pipe.VALU32, 1, (1, 1, 1), True),
    0x121: ("v_sub_co_ci_u32", Pipe.VALU32, 1, (1, 1, 1), True),
    0x122: ("v_subrev_co_ci_u32", Pipe.VALU32, 1, (1, 1, 1), True),
    0x209: ("v_fma_dx9_zero_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x20A: ("v_mad_i32_i24", Pipe.VALU32, 1, (1, 1, 1), False),
    0x20B: ("v_mad_u32_u24", Pipe.VALU32, 1, (1, 1, 1), False),
    0x210: ("v_bfe_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x211: ("v_bfe_i32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x212: ("v_bfi_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x213: ("v_fma_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x214: ("v_fma_f64", Pipe.VALUB_16, 2, (2, 2, 2), False),
    0x216: ("v_alignbit_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x217: ("v_alignbyte_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x219: ("v_min3_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21A: ("v_min3_i32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21B: ("v_min3_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21C: ("v_max3_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21D: ("v_max3_i32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21E: ("v_max3_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x21F: ("v_med3_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x220: ("v_med3_i32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x221: ("v_med3_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x240: ("v_xor3_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x244: ("v_perm_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x246: ("v_lshl_add_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x247: ("v_add_lshl_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x248: ("v_fma_f16", Pipe.VALU32, 1, (1, 1, 1), False),
    0x255: ("v_add3_u32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x256: ("v_lshl_or_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x257: ("v_and_or_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x258: ("v_or3_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x25B: ("v_permlane16_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x25C: ("v_permlanex16_b32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x2FC: ("v_div_scale_f32", Pipe.VALU32, 1, (1, 1, 1), True),
    0x2FD: ("v_div_scale_f64", Pipe.VALUB_16, 2, (2, 2, 2), True),
    0x2FE: ("v_mad_u64_u32", Pipe.VALUB_4, 2, (1, 1, 2), True),
    0x2FF: ("v_mad_i64_i32", Pipe.VALUB_4, 2, (1, 1, 2), True),
    0x300: ("v_add_co_u32", Pipe.VALU32, 1, (1, 1, 0), True),
    0x301: ("v_sub_co_u32", Pipe.VALU32, 1, (1, 1, 0), True),
    0x302: ("v_subrev_co_u32", Pipe.VALU32, 1, (1, 1, 0), True),
    0x327: ("v_add_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x328: ("v_mul_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x329: ("v_min_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x32A: ("v_max_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x32B: ("v_ldexp_f64", Pipe.VALUB_16, 2, (2, 1, 0), False),
    0x32C: ("v_mul_lo_u32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x32D: ("v_mul_hi_u32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x32E: ("v_mul_hi_i32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x33C: ("v_lshlrev_b64", Pipe.VALUB_2, 2, (1, 2, 0), False),
    0x33D: ("v_lshrrev_b64", Pipe.VALUB_2, 2, (1, 2, 0), False),
    0x33E: ("v_ashrrev_i64", Pipe.VALUB_2, 2, (1, 2, 0), False),
}

# VOP3P opcode -> (mnemonic, Pipe, dst_dw, (src0_dw, src1_dw, src2_dw))
_VOP3P_GFX11: Dict[int, Tuple[str, Pipe, int, Tuple[int, int, int]]] = {
    0x00: ("v_pk_mad_i16", Pipe.VALU32, 1, (1, 1, 1)),
    0x01: ("v_pk_mul_lo_u16", Pipe.VALU32, 1, (1, 1, 0)),
    0x02: ("v_pk_add_i16", Pipe.VALU32, 1, (1, 1, 0)),
    0x03: ("v_pk_sub_i16", Pipe.VALU32, 1, (1, 1, 0)),
    0x04: ("v_pk_lshlrev_b16", Pipe.VALU32, 1, (1, 1, 0)),
    0x05: ("v_pk_lshrrev_b16", Pipe.VALU32, 1, (1, 1, 0)),
    0x06: ("v_pk_ashrrev_i16", Pipe.VALU32, 1, (1, 1, 0)),
    0x07: ("v_pk_max_i16", Pipe.VALU32, 1, (1, 1, 0)),
    0x08: ("v_pk_min_i16", Pipe.VALU32, 1, (1, 1, 0)),
    0x09: ("v_pk_mad_u16", Pipe.VALU32, 1, (1, 1, 1)),
    0x0A: ("v_pk_add_u16", Pipe.VALU32, 1, (1, 1, 0)),
    0x0B: ("v_pk_sub_u16", Pipe.VALU32, 1, (1, 1, 0)),
    0x0C: ("v_pk_max_u16", Pipe.VALU32, 1, (1, 1, 0)),
    0x0D: ("v_pk_min_u16", Pipe.VALU32, 1, (1, 1, 0)),
    0x0E: ("v_pk_fma_f16", Pipe.VALU32, 1, (1, 1, 1)),
    0x0F: ("v_pk_add_f16", Pipe.VALU32, 1, (1, 1, 0)),
    0x10: ("v_pk_mul_f16", Pipe.VALU32, 1, (1, 1, 0)),
    0x11: ("v_pk_min_f16", Pipe.VALU32, 1, (1, 1, 0)),
    0x12: ("v_pk_max_f16", Pipe.VALU32, 1, (1, 1, 0)),
    0x13: ("v_dot2_f32_f16", Pipe.VALU32, 1, (1, 1, 1)),
    0x16: ("v_dot4_i32_iu8", Pipe.VALU32, 1, (1, 1, 1)),
    0x17: ("v_dot4_u32_u8", Pipe.VALU32, 1, (1, 1, 1)),
    0x18: ("v_dot8_i32_iu4", Pipe.VALU32, 1, (1, 1, 1)),
    0x19: ("v_dot8_u32_u4", Pipe.VALU32, 1, (1, 1, 1)),
    0x1A: ("v_dot2_f32_bf16", Pipe.VALU32, 1, (1, 1, 1)),
    0x20: ("v_fma_mix_f32", Pipe.VALU32, 1, (1, 1, 1)),
    0x21: ("v_fma_mixlo_f16", Pipe.VALU32, 1, (1, 1, 1)),
    0x22: ("v_fma_mixhi_f16", Pipe.VALU32, 1, (1, 1, 1)),
    0x40: ("v_wmma_f32_16x16x16_f16", Pipe.VALU_WMMA, 8, (8, 8, 8)),
    0x41: ("v_wmma_f32_16x16x16_bf16", Pipe.VALU_WMMA, 8, (8, 8, 8)),
    0x42: ("v_wmma_f16_16x16x16_f16", Pipe.VALU_WMMA, 8, (8, 8, 8)),
    0x43: ("v_wmma_bf16_16x16x16_bf16", Pipe.VALU_WMMA, 8, (8, 8, 8)),
    0x44: ("v_wmma_i32_16x16x16_iu8", Pipe.VALU_WMMA, 8, (4, 4, 8)),
    0x45: ("v_wmma_i32_16x16x16_iu4", Pipe.VALU_WMMA, 8, (2, 2, 8)),
}

_VOPD_OPS: Dict[int, str] = {
    0x00: "v_dual_fmac_f32",
    0x01: "v_dual_fmaak_f32",
    0x02: "v_dual_fmamk_f32",
    0x03: "v_dual_mul_f32",
    0x04: "v_dual_add_f32",
    0x05: "v_dual_sub_f32",
    0x06: "v_dual_subrev_f32",
    0x07: "v_dual_mul_dx9_zero_f32",
    0x08: "v_dual_mov_b32",
    0x09: "v_dual_cndmask_b32",
    0x0A: "v_dual_max_f32",
    0x0B: "v_dual_min_f32",
    0x0C: "v_dual_dot2acc_f32_f16",
    0x10: "v_dual_add_nc_u32",
    0x11: "v_dual_lshlrev_b32",
    0x12: "v_dual_and_b32",
}
_VOPD_LITERAL_OPS: frozenset[int] = frozenset({0x01, 0x02})

# DS opcode -> (mnemonic, is_load, dwords, is_2addr, addr_stride_bytes)
_DS_GFX11: Dict[int, Tuple[str, bool, int, bool, int]] = {
    0x00: ("ds_add_u32", False, 1, False, 0),
    0x01: ("ds_sub_u32", False, 1, False, 0),
    0x0D: ("ds_store_b32", False, 1, False, 0),
    0x0E: ("ds_store_2addr_b32", False, 2, True, 4),
    0x0F: ("ds_store_2addr_stride64_b32", False, 2, True, 256),
    0x15: ("ds_add_f32", False, 1, False, 0),
    0x1E: ("ds_store_b8", False, 1, False, 0),
    0x1F: ("ds_store_b16", False, 1, False, 0),
    0x35: ("ds_swizzle_b32", True, 1, False, 0),
    0x36: ("ds_load_b32", True, 1, False, 0),
    0x37: ("ds_load_2addr_b32", True, 2, True, 4),
    0x38: ("ds_load_2addr_stride64_b32", True, 2, True, 256),
    0x39: ("ds_load_i8", True, 1, False, 0),
    0x3A: ("ds_load_u8", True, 1, False, 0),
    0x3B: ("ds_load_i16", True, 1, False, 0),
    0x3C: ("ds_load_u16", True, 1, False, 0),
    0x4D: ("ds_store_b64", False, 2, False, 0),
    0x4E: ("ds_store_2addr_b64", False, 4, True, 8),
    0x4F: ("ds_store_2addr_stride64_b64", False, 4, True, 512),
    0x76: ("ds_load_b64", True, 2, False, 0),
    0x77: ("ds_load_2addr_b64", True, 4, True, 8),
    0x78: ("ds_load_2addr_stride64_b64", True, 4, True, 512),
    0xB0: ("ds_store_addtid_b32", False, 1, False, 0),
    0xB1: ("ds_load_addtid_b32", True, 1, False, 0),
    0xB2: ("ds_permute_b32", True, 1, False, 0),
    0xB3: ("ds_bpermute_b32", True, 1, False, 0),
    0xDE: ("ds_store_b96", False, 3, False, 0),
    0xDF: ("ds_store_b128", False, 4, False, 0),
    0xFE: ("ds_load_b96", True, 3, False, 0),
    0xFF: ("ds_load_b128", True, 4, False, 0),
}

_SMEM_GFX11: Dict[int, Tuple[str, int]] = {
    0x00: ("s_load_b32", 1),
    0x01: ("s_load_b64", 2),
    0x02: ("s_load_b128", 4),
    0x03: ("s_load_b256", 8),
    0x04: ("s_load_b512", 16),
    0x08: ("s_buffer_load_b32", 1),
    0x09: ("s_buffer_load_b64", 2),
    0x0A: ("s_buffer_load_b128", 4),
    0x0B: ("s_buffer_load_b256", 8),
    0x0C: ("s_buffer_load_b512", 16),
    0x20: ("s_gl1_inv", 0),
    0x21: ("s_dcache_inv", 0),
}

_FLAT_GFX11: Dict[int, Tuple[str, bool, int]] = {
    0x10: ("load_u8", True, 1),
    0x11: ("load_i8", True, 1),
    0x12: ("load_u16", True, 1),
    0x13: ("load_i16", True, 1),
    0x14: ("load_b32", True, 1),
    0x15: ("load_b64", True, 2),
    0x16: ("load_b96", True, 3),
    0x17: ("load_b128", True, 4),
    0x18: ("store_b8", False, 1),
    0x19: ("store_b16", False, 1),
    0x1A: ("store_b32", False, 1),
    0x1B: ("store_b64", False, 2),
    0x1C: ("store_b96", False, 3),
    0x1D: ("store_b128", False, 4),
    0x33: ("atomic_swap_b32", True, 1),
    0x35: ("atomic_add_u32", True, 1),
    0x56: ("atomic_add_f32", True, 1),
}

# MUBUF opcode -> (mnemonic, is_load, dwords)
_MUBUF_GFX11: Dict[int, Tuple[str, bool, int]] = {
    0x00: ("buffer_load_format_x", True, 1),
    0x01: ("buffer_load_format_xy", True, 2),
    0x02: ("buffer_load_format_xyz", True, 3),
    0x03: ("buffer_load_format_xyzw", True, 4),
    0x04: ("buffer_store_format_x", False, 1),
    0x05: ("buffer_store_format_xy", False, 2),
    0x06: ("buffer_store_format_xyz", False, 3),
    0x07: ("buffer_store_format_xyzw", False, 4),
    0x10: ("buffer_load_u8", True, 1),
    0x11: ("buffer_load_i8", True, 1),
    0x12: ("buffer_load_u16", True, 1),
    0x13: ("buffer_load_i16", True, 1),
    0x14: ("buffer_load_b32", True, 1),
    0x15: ("buffer_load_b64", True, 2),
    0x16: ("buffer_load_b96", True, 3),
    0x17: ("buffer_load_b128", True, 4),
    0x18: ("buffer_store_b8", False, 1),
    0x19: ("buffer_store_b16", False, 1),
    0x1A: ("buffer_store_b32", False, 1),
    0x1B: ("buffer_store_b64", False, 2),
    0x1C: ("buffer_store_b96", False, 3),
    0x1D: ("buffer_store_b128", False, 4),
    0x2B: ("buffer_gl0_inv", True, 0),
    0x2C: ("buffer_gl1_inv", True, 0),
    0x2D: ("buffer_load_lds_u8", True, 1),
    0x2E: ("buffer_load_lds_i8", True, 1),
    0x2F: ("buffer_load_lds_u16", True, 1),
    0x30: ("buffer_load_lds_i16", True, 1),
    0x31: ("buffer_load_lds_b32", True, 1),
    0x32: ("buffer_load_lds_format_x", True, 1),
    0x33: ("buffer_atomic_swap_b32", True, 1),
    0x35: ("buffer_atomic_add_u32", True, 1),
    0x56: ("buffer_atomic_add_f32", True, 1),
}

_MTBUF_GFX11: Dict[int, Tuple[str, bool, int]] = {
    0x00: ("tbuffer_load_format_x", True, 1),
    0x01: ("tbuffer_load_format_xy", True, 2),
    0x02: ("tbuffer_load_format_xyz", True, 3),
    0x03: ("tbuffer_load_format_xyzw", True, 4),
    0x04: ("tbuffer_store_format_x", False, 1),
    0x05: ("tbuffer_store_format_xy", False, 2),
    0x06: ("tbuffer_store_format_xyz", False, 3),
    0x07: ("tbuffer_store_format_xyzw", False, 4),
}

_VINTERP_GFX11: Dict[int, str] = {
    0x00: "v_interp_p10_f32",
    0x01: "v_interp_p2_f32",
    0x02: "v_interp_p10_f16_f32",
    0x03: "v_interp_p2_f16_f32",
    0x04: "v_interp_p10_rtz_f16_f32",
    0x05: "v_interp_p2_rtz_f16_f32",
}

_LDSDIR_GFX11: Dict[int, str] = {
    0x00: "lds_param_load",
    0x01: "lds_direct_load",
}


def _sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val & (sign - 1)) - (val & sign)


def _decode_reg_operand(code: int, dwords: int = 1, allow_vgpr: bool = True) -> Tuple[str, ...]:
    if dwords <= 0:
        return ()
    if allow_vgpr and 256 <= code <= 511:
        base = code - 256
        return tuple(f"v{base + i}" for i in range(dwords))
    if 0 <= code <= 105:
        return tuple(f"s{code + i}" for i in range(dwords))
    if code == VCC_LO_ENC:
        return ("vcc_lo",) if dwords == 1 else ("vcc_lo", "vcc_hi")
    if code == VCC_HI_ENC:
        return ("vcc_hi",)
    if code in (EXEC_LO_ENC, EXEC_HI_ENC):
        return (EXEC_REG,)
    return ()


def _decode_inline_u32(code: int) -> Optional[int]:
    if 128 <= code <= 192:
        return code - 128
    if 193 <= code <= 208:
        return (-(code - 192)) & 0xFFFFFFFF
    # Inline float constants (IEEE-754 f32)
    fconst = {
        240: 0x3F000000,  # 0.5
        241: 0xBF000000,  # -0.5
        242: 0x3F800000,  # 1.0
        243: 0xBF800000,  # -1.0
        244: 0x40000000,  # 2.0
        245: 0xC0000000,  # -2.0
        246: 0x40800000,  # 4.0
        247: 0xC0800000,  # -4.0
        248: 0x3E22F983,  # 1/(2*pi)
    }
    if code in fconst:
        return fconst[code]
    return None


def _dedup_regs(regs: Iterable[str]) -> Tuple[str, ...]:
    seen: Set[str] = set()
    out: List[str] = []
    for r in regs:
        cr = _canon_reg(r) if r else ""
        if cr and cr not in seen:
            seen.add(cr)
            out.append(cr)
    return tuple(out)


def _normalize_bytes(data: Union[bytes, bytearray, memoryview, str]) -> bytes:
    if isinstance(data, str):
        cleaned = data.replace("0x", "").replace("0X", "").replace(":", "").replace("_", "")
        cleaned = "".join(cleaned.split())
        if len(cleaned) % 2 != 0:
            raise ValueError(f"Hex string must have even number of hex digits, got {len(cleaned)}")
        return bytes.fromhex(cleaned)
    return bytes(data)


@dataclass(frozen=True)
class DecodedInsn:
    """Hardware-decoded GFX11 (RDNA3 Wave32) instruction."""

    pc: int
    size: int
    raw_words: Tuple[int, ...]
    encoding: str
    mnemonic: str
    pipe: Pipe
    dst: Tuple[str, ...] = ()
    src: Tuple[str, ...] = ()
    literal: Optional[int] = None
    simm16: Optional[int] = None
    branch_target: Optional[int] = None
    vmcnt: Optional[int] = None
    lgkmcnt: Optional[int] = None
    expcnt: Optional[int] = None
    vscnt: Optional[int] = None
    lds_dwords: int = 1
    lds_addr_reg: Optional[str] = None
    lds_offset: int = 0
    lds_offset0: int = 0
    lds_offset1: int = 0
    lds_is_2addr: bool = False
    lds_2addr_stride: int = 0
    exec_mask: Optional[int] = None
    is_vopd: bool = False
    vopd_x_mnemonic: str = ""
    vopd_y_mnemonic: str = ""
    vopd_vdstx: Optional[int] = None
    vopd_vdsty: Optional[int] = None
    vopd_bank_ok: bool = True
    writes_exec: bool = False
    # Structured operand codes for real Wave32 ISA execution
    op_code: int = 0
    src_codes: Tuple[int, ...] = ()
    dst_code: int = 0
    sdst_code: int = 0
    delay_instid0: int = 0
    delay_instskip: int = 0
    delay_instid1: int = 0

    def to_instruction(
        self,
        vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
        taken: bool = False,
        forwarded: bool = False,
        barrier_id: int = 0,
    ) -> Instruction:
        """Convert decoded GFX11 instruction into a simulator `Instruction`."""
        lds_addrs: Optional[Tuple[int, ...]] = None
        lds_sub_offsets: Optional[Tuple[int, ...]] = None
        if self.pipe in LDS_PIPES:
            if vgpr_state is not None and self.lds_addr_reg and self.lds_addr_reg in vgpr_state:
                base_lanes = vgpr_state[self.lds_addr_reg]
                if len(base_lanes) != WAVE_LANES:
                    raise ValueError(f"VGPR {self.lds_addr_reg} must have {WAVE_LANES} lanes")
                base_tuple = tuple(int(a) for a in base_lanes)
            else:
                base_tuple = tuple(lane * 4 for lane in range(WAVE_LANES))
            if self.lds_is_2addr:
                lds_addrs = base_tuple
                stride = self.lds_2addr_stride
                if self.lds_dwords == 2:
                    lds_sub_offsets = (self.lds_offset0 * stride, self.lds_offset1 * stride)
                else:
                    o0 = self.lds_offset0 * stride
                    o1 = self.lds_offset1 * stride
                    lds_sub_offsets = (o0, o0 + 4, o1, o1 + 4)
            else:
                lds_addrs = tuple(a + self.lds_offset for a in base_tuple)

        lgkm_t = self.lgkmcnt if self.lgkmcnt is not None else WAITCNT_LGKM_NONE
        vm_t = self.vmcnt if self.vmcnt is not None else WAITCNT_VM_NONE
        exp_t = self.expcnt if self.expcnt is not None else WAITCNT_EXP_NONE
        vs_t = self.vscnt if self.vscnt is not None else WAITCNT_VS_NONE
        return Instruction(
            pc=self.pc,
            pipe=self.pipe,
            dst=self.dst,
            src=self.src,
            taken=taken,
            forwarded=forwarded,
            lgkm_target=lgkm_t,
            vm_target=vm_t,
            exp_target=exp_t,
            vs_target=vs_t,
            barrier_id=barrier_id,
            exec_mask=self.exec_mask,
            lds_addrs=lds_addrs,
            lds_dwords=self.lds_dwords,
            lds_sub_offsets=lds_sub_offsets,
            is_vopd=self.is_vopd,
            mnemonic=self.mnemonic,
            decoded=self,
        )


def decode_gfx11(
    code_bytes: Union[bytes, bytearray, memoryview, str],
    base_pc: int = 0,
    stop_at_code_end: bool = False,
) -> List[DecodedInsn]:
    """Decode raw little-endian 32-bit/64-bit GFX11 (RDNA3 Wave32) machine code.

    Accepts raw bytes, hex string, or an AMDGPU ELF64 binary (auto-extracts `.text`).
    Zero external disassemblers required.
    """
    raw = _normalize_bytes(code_bytes)
    if raw[:4] == b"\x7fELF":
        kern = parse_amdgpu_elf(raw)
        raw = kern.text_bytes
        if base_pc == 0:
            base_pc = kern.text_addr
        stop_at_code_end = True
    if len(raw) % 4 != 0:
        raise ValueError(f"GFX11 code stream must be 4-byte aligned, got {len(raw)} bytes")

    def read_u32(offset: int) -> int:
        if offset + 4 > len(raw):
            raise ValueError(f"Truncated GFX11 instruction at byte offset 0x{offset:x}")
        return struct.unpack_from("<I", raw, offset)[0]

    def _has_dpp_or_lit(src_code: int) -> bool:
        return src_code in (LITERAL_ENC, DPP16_ENC, SDWA_ENC, DPP8_ENC, DPP8FI_ENC)

    out: List[DecodedInsn] = []
    off = 0
    n = len(raw)
    while off < n:
        pc = base_pc + off
        w0 = read_u32(off)
        top9 = (w0 >> 23) & 0x1FF
        top8 = (w0 >> 24) & 0xFF
        top7 = (w0 >> 25) & 0x7F
        top6 = (w0 >> 26) & 0x3F
        top4 = (w0 >> 28) & 0xF
        top2 = (w0 >> 30) & 0x3

        if top9 == 0x17F:  # SOPP
            op = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend(imm16, 16)
            mnem = _SOPP_GFX11.get(op, f"s_sopp_0x{op:02x}")
            if op == 0x1F and stop_at_code_end:
                break
            if op == 0x09:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.WAIT_LGKMCNT, simm16=simm16, op_code=op,
                    vmcnt=(imm16 >> 10) & 0x3F, lgkmcnt=(imm16 >> 4) & 0x3F, expcnt=imm16 & 0x7,
                ))
            elif op == 0x07:  # s_delay_alu
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.DELAY_ALU, simm16=simm16, op_code=op,
                    delay_instid0=imm16 & 0xF,
                    delay_instskip=(imm16 >> 4) & 0x7,
                    delay_instid1=(imm16 >> 7) & 0xF,
                ))
            elif op == 0x3D:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.BARRIER, simm16=simm16, op_code=op,
                ))
            elif op in (0x30, 0x31, 0x32):
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.ENDPGM, simm16=simm16, op_code=op,
                ))
            elif op in _SOPP_BRANCH_OPS:
                src_regs: Tuple[str, ...] = ()
                if op in (0x21, 0x22):
                    src_regs = ("scc",)
                elif op in (0x23, 0x24):
                    src_regs = ("vcc_lo",)
                elif op in (0x25, 0x26):
                    src_regs = (EXEC_REG,)
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.JUMP, src=src_regs, simm16=simm16, op_code=op,
                    branch_target=pc + 4 + simm16 * 4,
                ))
            else:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.SALU, simm16=simm16, op_code=op,
                ))
            off += 4
            continue

        if top9 == 0x17E:  # SOPC
            op = (w0 >> 16) & 0x7F
            ssrc1 = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            lit = read_u32(off + 4) if (ssrc0 == LITERAL_ENC or ssrc1 == LITERAL_ENC) else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _SOPC_GFX11.get(op, f"s_sopc_0x{op:02x}")
            dw = 2 if mnem.endswith(("_b64", "_u64", "_i64")) else 1
            src = _dedup_regs((*_decode_reg_operand(ssrc0, dw, False), *_decode_reg_operand(ssrc1, dw, False)))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="SOPC",
                mnemonic=mnem, pipe=Pipe.SALU, dst=("scc",), src=src, literal=lit,
                op_code=op, src_codes=(ssrc0, ssrc1),
            ))
            off += 4 * len(words)
            continue

        if top9 == 0x17D:  # SOP1
            sdst = (w0 >> 16) & 0x7F
            op = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            lit = read_u32(off + 4) if ssrc0 == LITERAL_ENC else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _SOP1_GFX11.get(op, f"s_sop1_0x{op:02x}")
            dw = 2 if mnem.endswith("_b64") else 1
            writes_exec = (sdst in (EXEC_LO_ENC, EXEC_HI_ENC)) or (op in _SOP1_SAVEEXEC_OPS)
            dst_list = list(_decode_reg_operand(sdst, dw, False))
            if writes_exec and EXEC_REG not in dst_list:
                dst_list.append(EXEC_REG)
            src_list = list(_decode_reg_operand(ssrc0, dw, False))
            if op in _SOP1_SAVEEXEC_OPS and EXEC_REG not in src_list:
                src_list.append(EXEC_REG)
            exec_mask: Optional[int] = None
            if sdst == EXEC_LO_ENC and op == 0x00:
                if lit is not None:
                    exec_mask = lit & FULL_EXEC_MASK
                else:
                    inl = _decode_inline_u32(ssrc0)
                    if inl is not None:
                        exec_mask = inl & FULL_EXEC_MASK
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="SOP1",
                mnemonic=mnem, pipe=Pipe.SALU_WR_EXEC if writes_exec else Pipe.SALU,
                dst=_dedup_regs(dst_list), src=_dedup_regs(src_list),
                literal=lit, exec_mask=exec_mask, writes_exec=writes_exec,
                op_code=op, dst_code=sdst, src_codes=(ssrc0,),
            ))
            off += 4 * len(words)
            continue

        if top4 == 0xB:  # SOPK
            op = (w0 >> 23) & 0x1F
            sdst = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend(imm16, 16)
            lit = read_u32(off + 4) if op == 0x13 else None  # s_setreg_imm32_b32
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _SOPK_GFX11.get(op, f"s_sopk_0x{op:02x}")
            if op in _SOPK_WAITCNT_OPS:
                # GFX11 SOPK s_waitcnt_{vscnt,vmcnt,expcnt,lgkmcnt}
                cnt = imm16 & 0x3F
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="SOPK",
                    mnemonic=mnem, pipe=Pipe.WAIT_LGKMCNT,
                    src=_decode_reg_operand(sdst, 1, False),
                    simm16=simm16, op_code=op, dst_code=sdst,
                    vscnt=cnt if op == 0x18 else None,
                    vmcnt=cnt if op == 0x19 else None,
                    expcnt=(imm16 & 0x7) if op == 0x1A else None,
                    lgkmcnt=cnt if op == 0x1B else None,
                ))
            elif op == 0x14:  # s_call_b64
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="SOPK",
                    mnemonic=mnem, pipe=Pipe.JUMP,
                    dst=_decode_reg_operand(sdst, 2, False),
                    simm16=simm16, branch_target=pc + 4 + simm16 * 4,
                    op_code=op, dst_code=sdst,
                ))
            elif op in _SOPK_CMP_OPS:
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="SOPK",
                    mnemonic=mnem, pipe=Pipe.SALU,
                    dst=("scc",), src=_decode_reg_operand(sdst, 1, False),
                    simm16=simm16, op_code=op, dst_code=sdst,
                ))
            else:
                writes_exec = sdst in (EXEC_LO_ENC, EXEC_HI_ENC)
                dst = _decode_reg_operand(sdst, 1, False)
                src = dst if op in (0x0F, 0x10) else ()
                exec_mask = (imm16 & FULL_EXEC_MASK) if (sdst == EXEC_LO_ENC and op == 0x00) else None
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="SOPK",
                    mnemonic=mnem, pipe=Pipe.SALU_WR_EXEC if writes_exec else Pipe.SALU,
                    dst=dst, src=src, literal=lit, simm16=simm16,
                    exec_mask=exec_mask, writes_exec=writes_exec,
                    op_code=op, dst_code=sdst,
                ))
            off += 4 * len(words)
            continue

        if top2 == 0x2:  # SOP2
            op = (w0 >> 23) & 0x7F
            sdst = (w0 >> 16) & 0x7F
            ssrc1 = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            lit = read_u32(off + 4) if (ssrc0 == LITERAL_ENC or ssrc1 == LITERAL_ENC) else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _SOP2_GFX11.get(op, f"s_sop2_0x{op:02x}")
            dw = 2 if mnem.endswith(("_b64", "_i64", "_u64")) else 1
            writes_exec = sdst in (EXEC_LO_ENC, EXEC_HI_ENC)
            dst_list = list(_decode_reg_operand(sdst, dw, False))
            if op in (0x00, 0x01, 0x02, 0x03, 0x04, 0x05):
                dst_list.append("scc")
            src_list = list(_decode_reg_operand(ssrc0, dw, False)) + list(_decode_reg_operand(ssrc1, dw, False))
            if op in (0x04, 0x05, 0x30, 0x31):
                src_list.append("scc")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="SOP2",
                mnemonic=mnem, pipe=Pipe.SALU_WR_EXEC if writes_exec else Pipe.SALU,
                dst=_dedup_regs(dst_list), src=_dedup_regs(src_list), literal=lit,
                writes_exec=writes_exec, op_code=op, dst_code=sdst, src_codes=(ssrc0, ssrc1),
            ))
            off += 4 * len(words)
            continue

        if top7 == 0x3F:  # VOP1
            vdst = (w0 >> 17) & 0xFF
            op = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            extra = read_u32(off + 4) if _has_dpp_or_lit(src0) else None
            lit = extra if src0 == LITERAL_ENC else None
            words = (w0, extra) if extra is not None else (w0,)
            mnem = _VOP1_GFX11.get(op, f"v_vop1_0x{op:02x}")
            if op in _VOP1_VALUT_OPS:
                pipe = Pipe.VALUT_4
            elif op in _VOP1_F64_OPS:
                pipe = Pipe.VALUB_16
            else:
                pipe = Pipe.VALU32
            dst_dw = 2 if mnem.endswith("_f64") and "_f32_f64" not in mnem and "_i32_f64" not in mnem and "_u32_f64" not in mnem else 1
            src_dw = 2 if "_f64" in mnem and not mnem.endswith(("_f64_i32", "_f64_u32", "_f64_f32")) else 1
            dst = () if op == 0x00 else ((f"s{vdst}",) if op == 0x02 else tuple(f"v{vdst + i}" for i in range(dst_dw)))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP1",
                mnemonic=mnem, pipe=pipe, dst=dst,
                src=_decode_reg_operand(src0, src_dw, True), literal=lit,
                op_code=op, dst_code=vdst, src_codes=(src0,),
            ))
            off += 4 * len(words)
            continue

        if top7 == 0x3E:  # VOPC (op < 0x80 -> vcc_lo) / VOPCX (op >= 0x80 -> exec_lo)
            op = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            extra = read_u32(off + 4) if _has_dpp_or_lit(src0) else None
            lit = extra if src0 == LITERAL_ENC else None
            words = (w0, extra) if extra is not None else (w0,)
            is_vopcx = (op >= 0x80)
            mnem = _VOPC_GFX11.get(op, f"v_cmpx_0x{op:02x}_e32" if is_vopcx else f"v_cmp_0x{op:02x}_e32")
            dw = 2 if any(t in mnem for t in ("_f64_", "_i64_", "_u64_")) else 1
            src = _dedup_regs((*_decode_reg_operand(src0, dw, True), *(f"v{vsrc1 + i}" for i in range(dw))))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words,
                encoding="VOPCX" if is_vopcx else "VOPC",
                mnemonic=mnem,
                pipe=Pipe.VALU1_WR_EXEC if is_vopcx else Pipe.VALU32,
                dst=(EXEC_REG,) if is_vopcx else ("vcc_lo",),
                src=src, literal=lit, writes_exec=is_vopcx,
                op_code=op, dst_code=EXEC_LO_ENC if is_vopcx else VCC_LO_ENC,
                src_codes=(src0, 256 + vsrc1),
            ))
            off += 4 * len(words)
            continue

        if (w0 >> 31) == 0:  # VOP2
            op = (w0 >> 25) & 0x3F
            vdst = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            has_lit = (op in _VOP2_LITERAL_OPS) or (src0 == LITERAL_ENC)
            has_dpp = (not has_lit) and _has_dpp_or_lit(src0)
            extra = read_u32(off + 4) if (has_lit or has_dpp) else None
            lit = extra if has_lit else None
            words = (w0, extra) if extra is not None else (w0,)
            mnem = _VOP2_GFX11.get(op, f"v_vop2_0x{op:02x}")
            src_list = list(_decode_reg_operand(src0, 1, True)) + [f"v{vsrc1}"]
            if op in (0x01, 0x20, 0x21, 0x22):
                src_list.append("vcc_lo")
            if op in (0x02, 0x06, 0x2B, 0x36):
                src_list.append(f"v{vdst}")
            dst_list = [f"v{vdst}"]
            if op in (0x20, 0x21, 0x22):
                dst_list.append("vcc_lo")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP2",
                mnemonic=mnem, pipe=Pipe.VALU32, dst=_dedup_regs(dst_list),
                src=_dedup_regs(src_list), literal=lit,
                op_code=op, dst_code=vdst, src_codes=(src0, 256 + vsrc1),
            ))
            off += 4 * len(words)
            continue

        if top6 == 0x32:  # VOPD (64-bit + optional 32b lit)
            w1 = read_u32(off + 4)
            opx = (w0 >> 22) & 0xF
            opy = (w0 >> 17) & 0x1F
            vsrcx1 = (w0 >> 9) & 0xFF
            srcx0 = w0 & 0x1FF
            vdstx = (w1 >> 24) & 0xFF
            vdsty = (((w1 >> 17) & 0x7F) << 1) | ((vdstx & 1) ^ 1)
            vsrcy1 = (w1 >> 9) & 0xFF
            srcy0 = w1 & 0x1FF
            has_lit = (
                (opx in _VOPD_LITERAL_OPS) or (opy in _VOPD_LITERAL_OPS)
                or (srcx0 == LITERAL_ENC) or (srcy0 == LITERAL_ENC)
            )
            lit = read_u32(off + 8) if has_lit else None
            words = (w0, w1, lit) if lit is not None else (w0, w1)
            mx = _VOPD_OPS.get(opx, f"v_dual_x_0x{opx:02x}")
            my = _VOPD_OPS.get(opy, f"v_dual_y_0x{opy:02x}")
            src_list: List[str] = list(_decode_reg_operand(srcx0, 1, True)) + list(_decode_reg_operand(srcy0, 1, True))
            if opx != 0x08:
                src_list.append(f"v{vsrcx1}")
            if opy != 0x08:
                src_list.append(f"v{vsrcy1}")
            if opx in (0x00, 0x0C):
                src_list.append(f"v{vdstx}")
            if opy in (0x00, 0x0C):
                src_list.append(f"v{vdsty}")
            if opx == 0x09 or opy == 0x09:
                src_list.append("vcc_lo")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOPD",
                mnemonic=f"{mx} :: {my}", pipe=Pipe.VALU32,
                dst=_dedup_regs((f"v{vdstx}", f"v{vdsty}")),
                src=_dedup_regs(src_list), literal=lit,
                is_vopd=True, vopd_x_mnemonic=mx, vopd_y_mnemonic=my,
                vopd_vdstx=vdstx, vopd_vdsty=vdsty,
                vopd_bank_ok=((vdstx & 1) != (vdsty & 1)),
                op_code=(opx << 8) | opy,
                dst_code=(vdstx << 8) | vdsty,
                src_codes=(srcx0, 256 + vsrcx1, srcy0, 256 + vsrcy1),
            ))
            off += 4 * len(words)
            continue

        if top8 == 0xCC:  # VOP3P (64-bit + optional 32b lit: packed 16b, dot, fma_mix, WMMA)
            w1 = read_u32(off + 4)
            op = (w0 >> 16) & 0x7F
            vdst = w0 & 0xFF
            src0 = w1 & 0x1FF
            src1 = (w1 >> 9) & 0x1FF
            src2 = (w1 >> 18) & 0x1FF
            has_lit = (src0 == LITERAL_ENC or src1 == LITERAL_ENC or src2 == LITERAL_ENC)
            lit = read_u32(off + 8) if has_lit else None
            words = (w0, w1, lit) if lit is not None else (w0, w1)
            meta = _VOP3P_GFX11.get(op)
            if meta is not None:
                mnem, pipe, dst_dw, (s0_dw, s1_dw, s2_dw) = meta
            else:
                mnem = f"v_vop3p_0x{op:02x}"
                pipe = Pipe.VALU_WMMA if op >= 0x40 else Pipe.VALU32
                dst_dw, s0_dw, s1_dw, s2_dw = (8, 8, 8, 8) if op >= 0x40 else (1, 1, 1, 1)
            dst_regs = tuple(f"v{vdst + i}" for i in range(dst_dw))
            src_list = (
                list(_decode_reg_operand(src0, s0_dw, True))
                + list(_decode_reg_operand(src1, s1_dw, True))
                + list(_decode_reg_operand(src2, s2_dw, True))
            )
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP3P",
                mnemonic=mnem, pipe=pipe, dst=dst_regs, src=_dedup_regs(src_list),
                literal=lit, op_code=op, dst_code=vdst, src_codes=(src0, src1, src2),
            ))
            off += 4 * len(words)
            continue

        if top8 == 0xCD:  # VINTERP (64-bit + optional 32b lit)
            w1 = read_u32(off + 4)
            op = (w0 >> 16) & 0x7F
            vdst = w0 & 0xFF
            src0 = w1 & 0x1FF
            src1 = (w1 >> 9) & 0x1FF
            src2 = (w1 >> 18) & 0x1FF
            has_lit = (src0 == LITERAL_ENC or src1 == LITERAL_ENC or src2 == LITERAL_ENC)
            lit = read_u32(off + 8) if has_lit else None
            words = (w0, w1, lit) if lit is not None else (w0, w1)
            mnem = _VINTERP_GFX11.get(op, f"v_interp_0x{op:02x}")
            src_list = (
                list(_decode_reg_operand(src0, 1, True))
                + list(_decode_reg_operand(src1, 1, True))
                + list(_decode_reg_operand(src2, 1, True))
            )
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VINTERP",
                mnemonic=mnem, pipe=Pipe.VINTERP, dst=(f"v{vdst}",),
                src=_dedup_regs(src_list), literal=lit,
                op_code=op, dst_code=vdst, src_codes=(src0, src1, src2),
            ))
            off += 4 * len(words)
            continue

        if top8 == 0xCE:  # LDSDIR (32-bit)
            op = (w0 >> 20) & 0x3
            vdst = w0 & 0xFF
            mnem = _LDSDIR_GFX11.get(op, f"lds_dir_0x{op:02x}")
            out.append(DecodedInsn(
                pc=pc, size=4, raw_words=(w0,), encoding="LDSDIR",
                mnemonic=mnem, pipe=Pipe.LDS_RD, dst=(f"v{vdst}",),
                op_code=op, dst_code=vdst,
            ))
            off += 4
            continue

        if top6 == 0x35:  # VOP3 / VOP3SD (64-bit + optional 32b lit)
            w1 = read_u32(off + 4)
            op = (w0 >> 16) & 0x3FF
            sdst = (w0 >> 8) & 0x7F
            vdst = w0 & 0xFF
            src0 = w1 & 0x1FF
            src1 = (w1 >> 9) & 0x1FF
            src2 = (w1 >> 18) & 0x1FF
            has_lit = (src0 == LITERAL_ENC or src1 == LITERAL_ENC or src2 == LITERAL_ENC)
            lit = read_u32(off + 8) if has_lit else None
            words = (w0, w1, lit) if lit is not None else (w0, w1)

            if op <= 0x0FF:
                # VOP3-encoded VOPC (0x000..0x07F -> scalar dst in w0[7:0]) / VOPCX (0x080..0x0FF -> exec_lo)
                is_vopcx = (op >= 0x080)
                base_mnem = _VOPC_GFX11.get(op, f"v_cmpx_0x{op:02x}_e32" if is_vopcx else f"v_cmp_0x{op:02x}_e32")
                mnem = base_mnem[:-4] + "_e64" if base_mnem.endswith("_e32") else f"{base_mnem}_e64"
                dw = 2 if any(t in mnem for t in ("_f64_", "_i64_", "_u64_")) else 1
                dst_regs = (EXEC_REG,) if is_vopcx else _decode_reg_operand(vdst & 0x7F, 1, False)
                src_list = list(_decode_reg_operand(src0, dw, True)) + list(_decode_reg_operand(src1, dw, True))
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words,
                    encoding="VOP3_VOPCX" if is_vopcx else "VOP3_VOPC",
                    mnemonic=mnem,
                    pipe=Pipe.VALU1_WR_EXEC if is_vopcx else Pipe.VALU32,
                    dst=dst_regs, src=_dedup_regs(src_list), literal=lit,
                    writes_exec=is_vopcx, op_code=op,
                    dst_code=EXEC_LO_ENC if is_vopcx else (vdst & 0x7F),
                    src_codes=(src0, src1),
                ))
            elif op in _VOP3_GFX11:
                mnem, pipe, dst_dw, (s0_dw, s1_dw, s2_dw), writes_sdst = _VOP3_GFX11[op]
                dst_list = [f"v{vdst + i}" for i in range(dst_dw)]
                if writes_sdst:
                    dst_list.extend(_decode_reg_operand(sdst, 1, False))
                src_list = (
                    list(_decode_reg_operand(src0, s0_dw, True))
                    + list(_decode_reg_operand(src1, s1_dw, True))
                    + list(_decode_reg_operand(src2, s2_dw, True))
                )
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words,
                    encoding="VOP3SD" if writes_sdst else "VOP3",
                    mnemonic=mnem, pipe=pipe, dst=_dedup_regs(dst_list),
                    src=_dedup_regs(src_list), literal=lit,
                    op_code=op, dst_code=vdst, sdst_code=sdst, src_codes=(src0, src1, src2),
                ))
            elif 0x100 <= op <= 0x17F:
                # VOP2 in VOP3 (_e64) form
                sub = op - 0x100
                base_m = _VOP2_GFX11.get(sub, f"v_vop2_0x{sub:02x}")
                mnem = f"{base_m}_e64"
                s2_dw = 1 if sub in (0x01, 0x20, 0x21, 0x22) else 0
                src_list = (
                    list(_decode_reg_operand(src0, 1, True))
                    + list(_decode_reg_operand(src1, 1, True))
                    + list(_decode_reg_operand(src2, s2_dw, True))
                )
                if sub in (0x02, 0x06, 0x2B, 0x36):
                    src_list.append(f"v{vdst}")
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="VOP3",
                    mnemonic=mnem, pipe=Pipe.VALU32, dst=(f"v{vdst}",),
                    src=_dedup_regs(src_list), literal=lit,
                    op_code=op, dst_code=vdst, sdst_code=sdst, src_codes=(src0, src1, src2),
                ))
            elif 0x180 <= op <= 0x1FF:
                # VOP1 in VOP3 (_e64) form
                sub = op - 0x180
                base_m = _VOP1_GFX11.get(sub, f"v_vop1_0x{sub:02x}")
                mnem = f"{base_m}_e64"
                pipe = Pipe.VALUT_4 if sub in _VOP1_VALUT_OPS else (Pipe.VALUB_16 if sub in _VOP1_F64_OPS else Pipe.VALU32)
                dst = () if sub == 0x00 else ((f"s{vdst}",) if sub == 0x02 else (f"v{vdst}",))
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="VOP3",
                    mnemonic=mnem, pipe=pipe, dst=dst,
                    src=_decode_reg_operand(src0, 1, True), literal=lit,
                    op_code=op, dst_code=vdst, src_codes=(src0,),
                ))
            else:
                mnem = f"v_vop3_0x{op:03x}"
                src_list = (
                    list(_decode_reg_operand(src0, 1, True))
                    + list(_decode_reg_operand(src1, 1, True))
                    + list(_decode_reg_operand(src2, 1, True))
                )
                out.append(DecodedInsn(
                    pc=pc, size=4 * len(words), raw_words=words, encoding="VOP3",
                    mnemonic=mnem, pipe=Pipe.VALU32, dst=(f"v{vdst}",),
                    src=_dedup_regs(src_list), literal=lit,
                    op_code=op, dst_code=vdst, sdst_code=sdst, src_codes=(src0, src1, src2),
                ))
            off += 4 * len(words)
            continue

        if top6 == 0x36:  # DS (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            offset0 = w0 & 0xFF
            offset1 = (w0 >> 8) & 0xFF
            offset16 = w0 & 0xFFFF
            addr_v = w1 & 0xFF
            data0_v = (w1 >> 8) & 0xFF
            data1_v = (w1 >> 16) & 0xFF
            vdst = (w1 >> 24) & 0xFF
            ds_meta = _DS_GFX11.get(op)
            if ds_meta is not None:
                mnem, is_load, dwords, is_2addr, stride = ds_meta
            else:
                is_load = bool(op & 0x20)
                mnem, dwords, is_2addr, stride = f"ds_0x{op:02x}", 1, False, 0
            if is_load:
                pipe = Pipe.LDS_RD
                dst_regs = tuple(f"v{vdst + i}" for i in range(dwords))
                src_regs = (f"v{addr_v}",)
                if op in (0xB2, 0xB3):  # ds_permute / ds_bpermute also read data0
                    src_regs = _dedup_regs((f"v{addr_v}", f"v{data0_v}"))
            else:
                pipe = LDS_WR_BY_DWORDS[min(5, max(1, dwords))]
                dst_regs = ()
                src_list = [f"v{addr_v}"]
                if is_2addr:
                    half_dw = max(1, dwords // 2)
                    src_list.extend(f"v{data0_v + i}" for i in range(half_dw))
                    src_list.extend(f"v{data1_v + i}" for i in range(half_dw))
                else:
                    src_list.extend(f"v{data0_v + i}" for i in range(dwords))
                src_regs = _dedup_regs(src_list)
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="DS",
                mnemonic=mnem, pipe=pipe, dst=dst_regs, src=src_regs,
                lds_dwords=dwords, lds_addr_reg=f"v{addr_v}",
                lds_offset=0 if is_2addr else offset16,
                lds_offset0=offset0, lds_offset1=offset1,
                lds_is_2addr=is_2addr, lds_2addr_stride=stride,
                op_code=op, dst_code=vdst, src_codes=(addr_v, data0_v, data1_v),
            ))
            off += 8
            continue

        if top6 == 0x3D:  # SMEM (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            sdata = (w0 >> 6) & 0x7F
            sbase = (w0 & 0x3F) << 1
            soffset = (w1 >> 25) & 0x7F
            ioffset = _sign_extend(w1 & 0x1FFFFF, 21)
            mnem, dwords = _SMEM_GFX11.get(op, (f"s_smem_0x{op:02x}", 1))
            base_dw = 4 if 0x08 <= op <= 0x0C else 2
            dst_regs = _decode_reg_operand(sdata, dwords, False) if dwords > 0 else ()
            src_list = list(_decode_reg_operand(sbase, base_dw, False))
            if soffset != NULL_ENC:
                src_list.extend(_decode_reg_operand(soffset, 1, False))
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="SMEM",
                mnemonic=mnem, pipe=Pipe.SMEM_RD, dst=dst_regs, src=_dedup_regs(src_list),
                lds_offset=ioffset, op_code=op, dst_code=sdata, src_codes=(sbase, soffset),
            ))
            off += 8
            continue

        if top6 == 0x37:  # FLAT / SCRATCH / GLOBAL (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0x7F
            seg = (w0 >> 16) & 0x3
            offset = _sign_extend(w0 & 0x1FFF, 13)
            vaddr = w1 & 0xFF
            vdata = (w1 >> 8) & 0xFF
            saddr = (w1 >> 16) & 0x7F
            vdst = (w1 >> 24) & 0xFF
            prefix = {0: "flat", 1: "scratch", 2: "global"}.get(seg, "flat")
            fmeta = _FLAT_GFX11.get(op)
            if fmeta is not None:
                op_name, is_load, dwords = fmeta
                mnem = f"{prefix}_{op_name}"
            else:
                is_load = op < 0x18 or op >= 0x30
                mnem, dwords = f"{prefix}_0x{op:02x}", 1
            addr_dw = 1 if (seg != 0 and saddr != NULL_ENC) else 2
            src_list = [f"v{vaddr + i}" for i in range(addr_dw)]
            if saddr != NULL_ENC:
                src_list.extend(_decode_reg_operand(saddr, 2 if seg == 2 else 1, False))
            if is_load:
                pipe = Pipe.VMEM_RD
                dst_regs = tuple(f"v{vdst + i}" for i in range(dwords))
                if op >= 0x30:  # atomic also reads vdata
                    src_list.extend(f"v{vdata + i}" for i in range(dwords))
            else:
                pipe = Pipe.VMEM_WR
                dst_regs = ()
                src_list.extend(f"v{vdata + i}" for i in range(dwords))
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding=prefix.upper(),
                mnemonic=mnem, pipe=pipe, dst=dst_regs, src=_dedup_regs(src_list),
                lds_offset=offset, op_code=op, dst_code=vdst, src_codes=(vaddr, vdata, saddr),
            ))
            off += 8
            continue

        if top6 == 0x38:  # MUBUF (64-bit: buffer_load_b32 [0xE0500000]..b128, buffer_store_*, etc.)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            offset = w0 & 0xFFF
            vaddr = w1 & 0xFF
            vdata = (w1 >> 8) & 0xFF
            srsrc = ((w1 >> 16) & 0x1F) << 2
            offen = (w1 >> 22) & 1
            idxen = (w1 >> 23) & 1
            soffset = (w1 >> 24) & 0xFF
            mmeta = _MUBUF_GFX11.get(op)
            if mmeta is not None:
                mnem, is_load, dwords = mmeta
            else:
                is_load = not (0x04 <= op <= 0x07 or 0x0C <= op <= 0x0F or 0x18 <= op <= 0x1D)
                mnem, dwords = f"buffer_0x{op:02x}", 1
            src_list: List[str] = list(_decode_reg_operand(srsrc, 4, False))
            addr_dw = (1 if offen else 0) + (1 if idxen else 0)
            if addr_dw > 0:
                src_list.extend(f"v{vaddr + i}" for i in range(addr_dw))
            if soffset != NULL_ENC:
                src_list.extend(_decode_reg_operand(soffset, 1, False))
            if is_load:
                pipe = Pipe.VMEM_RD
                dst_regs = tuple(f"v{vdata + i}" for i in range(dwords)) if dwords > 0 else ()
                if op >= 0x33:  # atomic
                    src_list.extend(f"v{vdata + i}" for i in range(dwords))
            else:
                pipe = Pipe.VMEM_WR
                dst_regs = ()
                src_list.extend(f"v{vdata + i}" for i in range(dwords))
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="MUBUF",
                mnemonic=mnem, pipe=pipe, dst=dst_regs, src=_dedup_regs(src_list),
                lds_offset=offset, op_code=op, dst_code=vdata, src_codes=(vaddr, srsrc, soffset),
            ))
            off += 8
            continue

        if top6 == 0x3A:  # MTBUF (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 15) & 0xF
            offset = w0 & 0xFFF
            vaddr = w1 & 0xFF
            vdata = (w1 >> 8) & 0xFF
            srsrc = ((w1 >> 16) & 0x1F) << 2
            offen = (w1 >> 22) & 1
            idxen = (w1 >> 23) & 1
            soffset = (w1 >> 24) & 0xFF
            mnem, is_load, dwords = _MTBUF_GFX11.get(op, (f"tbuffer_0x{op:02x}", True, 1))
            src_list = list(_decode_reg_operand(srsrc, 4, False))
            addr_dw = (1 if offen else 0) + (1 if idxen else 0)
            if addr_dw > 0:
                src_list.extend(f"v{vaddr + i}" for i in range(addr_dw))
            if soffset != NULL_ENC:
                src_list.extend(_decode_reg_operand(soffset, 1, False))
            pipe = Pipe.VMEM_RD if is_load else Pipe.VMEM_WR
            dst_regs = tuple(f"v{vdata + i}" for i in range(dwords)) if is_load else ()
            if not is_load:
                src_list.extend(f"v{vdata + i}" for i in range(dwords))
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="MTBUF",
                mnemonic=mnem, pipe=pipe, dst=dst_regs, src=_dedup_regs(src_list),
                lds_offset=offset, op_code=op, dst_code=vdata, src_codes=(vaddr, srsrc, soffset),
            ))
            off += 8
            continue

        if top6 == 0x3C:  # MIMG (64-bit + optional NSA extra address words)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            nsa_dwords = (w0 >> 1) & 0x3
            extra_words = tuple(read_u32(off + 8 + 4 * i) for i in range(nsa_dwords))
            words = (w0, w1, *extra_words)
            vaddr = w1 & 0xFF
            vdata = (w1 >> 8) & 0xFF
            srsrc = ((w1 >> 16) & 0x1F) << 2
            dmask = (w0 >> 8) & 0xF
            dwords = max(1, bin(dmask).count("1")) if dmask else 1
            is_store = (0x08 <= op <= 0x0F)
            pipe = Pipe.VMEM_WR if is_store else Pipe.VMEM_RD
            src_list = [f"v{vaddr}", *_decode_reg_operand(srsrc, 8, False)]
            dst_regs = () if is_store else tuple(f"v{vdata + i}" for i in range(dwords))
            if is_store:
                src_list.extend(f"v{vdata + i}" for i in range(dwords))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="MIMG",
                mnemonic=f"image_0x{op:02x}", pipe=pipe, dst=dst_regs, src=_dedup_regs(src_list),
                op_code=op, dst_code=vdata, src_codes=(vaddr, srsrc),
            ))
            off += 4 * len(words)
            continue

        if top6 == 0x3E:  # EXP (64-bit)
            w1 = read_u32(off + 4)
            en = w0 & 0xF
            target = (w0 >> 4) & 0x3F
            src_list = [f"v{(w1 >> (8 * i)) & 0xFF}" for i in range(4) if (en & (1 << i))]
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="EXP",
                mnemonic=f"exp_target_{target}", pipe=Pipe.VMEM_WR, dst=(),
                src=_dedup_regs(src_list), op_code=target,
            ))
            off += 8
            continue

        raise ValueError(f"Unsupported GFX11 instruction word 0x{w0:08x} at byte offset 0x{off:x}")

    return out


# ---------------------------------------------------------------------------
# AMDGPU ELF64 Parser (`parse_amdgpu_elf`, `decode_elf`)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AMDGPUKernel:
    """Parsed AMDGPU ELF64 kernel object (`.text`, `.rodata` descriptor, symbol)."""

    name: str
    e_machine: int
    text_addr: int
    text_offset: int
    text_bytes: bytes
    lds_bytes: int = 0
    private_bytes: int = 0
    kernarg_bytes: int = 0
    wavefront_size32: bool = True
    rsrc1: int = 0
    rsrc2: int = 0
    rsrc3: int = 0
    sections: Mapping[str, Tuple[int, int, int]] = field(default_factory=dict)


def parse_amdgpu_elf(elf_bytes: Union[bytes, bytearray, memoryview]) -> AMDGPUKernel:
    """Parse an AMDGPU ELF64 binary (standard LLVM/lld or minimal HSA ELF) without external deps."""
    elf = bytes(elf_bytes)
    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        raise ValueError("Not a valid ELF binary (missing 0x7fELF magic)")
    if elf[4] != 2:
        raise ValueError(f"Expected 64-bit ELF (EI_CLASS=2), got {elf[4]}")
    if elf[5] not in (0, 1):
        raise ValueError(f"Expected little-endian ELF (EI_DATA=1 or 0), got {elf[5]}")

    e_machine = struct.unpack_from("<H", elf, 18)[0]
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 58)
    e_shentsize = e_shentsize or 64
    if e_shoff == 0 or e_shnum == 0 or e_shstrndx >= e_shnum:
        raise ValueError("ELF binary missing valid section header table")
    if e_shoff + e_shnum * e_shentsize > len(elf):
        raise ValueError("ELF section header table exceeds file length")

    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    _, _, _, _, shstr_off, shstr_sz = struct.unpack_from("<IIQQQQ", elf, shstr_hdr)
    shstr = elf[shstr_off:shstr_off + shstr_sz]

    sections: Dict[str, Tuple[int, int, int]] = {}
    sec_bytes: Dict[str, bytes] = {}
    for i in range(e_shnum):
        off_hdr = e_shoff + i * e_shentsize
        sh_name, _sh_type, _sh_flags, sh_addr, sh_off, sh_size = struct.unpack_from("<IIQQQQ", elf, off_hdr)
        nul = shstr.find(b"\x00", sh_name)
        name = shstr[sh_name:nul if nul != -1 else len(shstr)].decode("ascii", errors="replace")
        sections[name] = (sh_addr, sh_off, sh_size)
        if sh_off + sh_size <= len(elf):
            sec_bytes[name] = elf[sh_off:sh_off + sh_size]

    if ".text" not in sections or ".text" not in sec_bytes:
        raise ValueError("AMDGPU ELF missing .text section")
    text_addr, text_off, _ = sections[".text"]
    text_raw = sec_bytes[".text"]

    # Extract kernel symbol name from .dynsym/.dynstr or .symtab/.strtab
    kern_name = "kernel"
    for sym_sec, str_sec in ((".dynsym", ".dynstr"), (".symtab", ".strtab")):
        if sym_sec in sec_bytes and str_sec in sec_bytes:
            syms, strs = sec_bytes[sym_sec], sec_bytes[str_sec]
            for pos in range(0, len(syms) - 23, 24):
                st_name, st_info, _other, _shndx, _val, _sz = struct.unpack_from("<IBBHQQ", syms, pos)
                st_type = st_info & 0xF
                if st_name > 0 and st_type in (1, 2):  # STT_OBJECT (.kd) or STT_FUNC
                    nul = strs.find(b"\x00", st_name)
                    cand = strs[st_name:nul if nul != -1 else len(strs)].decode("ascii", errors="replace")
                    if cand:
                        kern_name = cand[:-3] if cand.endswith(".kd") else cand
                        break
            if kern_name != "kernel":
                break

    # Parse AMD HSA kernel_descriptor_t from .rodata when present
    lds_bytes = priv_bytes = kernarg_bytes = rsrc1 = rsrc2 = rsrc3 = 0
    wave32 = True
    if ".rodata" in sec_bytes and len(sec_bytes[".rodata"]) >= 64:
        ro = sec_bytes[".rodata"]
        lds_bytes, priv_bytes, kernarg_bytes = struct.unpack_from("<III", ro, 0)
        rsrc1, rsrc2 = struct.unpack_from("<II", ro, 48)
        kd_flags = struct.unpack_from("<H", ro, 56)[0]
        rsrc3 = struct.unpack_from("<I", ro, 60)[0]
        wave32 = bool((kd_flags >> 10) & 1) or True

    return AMDGPUKernel(
        name=kern_name,
        e_machine=e_machine,
        text_addr=text_addr,
        text_offset=text_off,
        text_bytes=text_raw,
        lds_bytes=lds_bytes,
        private_bytes=priv_bytes,
        kernarg_bytes=kernarg_bytes,
        wavefront_size32=wave32,
        rsrc1=rsrc1,
        rsrc2=rsrc2,
        rsrc3=rsrc3,
        sections=sections,
    )


def decode_elf(
    elf_bytes: Union[bytes, bytearray, memoryview],
    stop_at_code_end: bool = True,
) -> Tuple[AMDGPUKernel, List[DecodedInsn]]:
    """Parse an AMDGPU ELF64 binary and decode its GFX11 `.text` section."""
    kern = parse_amdgpu_elf(elf_bytes)
    insns = decode_gfx11(kern.text_bytes, base_pc=kern.text_addr, stop_at_code_end=stop_at_code_end)
    return kern, insns


# Hardware-exact GFX1100 Encoders

def encode_sopp(op: int, simm16: int = 0) -> bytes:
    return struct.pack("<I", (0x17F << 23) | ((op & 0x7F) << 16) | (simm16 & 0xFFFF))


def encode_s_waitcnt(
    vmcnt: int = WAITCNT_VM_NONE, lgkmcnt: int = WAITCNT_LGKM_NONE, expcnt: int = WAITCNT_EXP_NONE
) -> bytes:
    return encode_sopp(0x09, ((vmcnt & 0x3F) << 10) | ((lgkmcnt & 0x3F) << 4) | (expcnt & 0x7))


def encode_s_waitcnt_vscnt(vscnt: int = 0, sdst: int = NULL_ENC) -> bytes:
    """Encode GFX1100 SOPK `s_waitcnt_vscnt null, N` (0xBC7C0000 | N)."""
    return encode_sopk(0x18, sdst, vscnt & 0x3F)


def encode_s_barrier() -> bytes:
    return encode_sopp(0x3D, 0)


def encode_s_endpgm() -> bytes:
    return encode_sopp(0x30, 0)


def encode_s_branch(simm16: int, cond_op: int = 0x20) -> bytes:
    return encode_sopp(cond_op, simm16)


def encode_sop2(op: int, sdst: int, ssrc0: int, ssrc1: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x2 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_sop1(op: int, sdst: int, ssrc0: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x17D << 23) | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc0 & 0xFF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_sopc(op: int, ssrc0: int, ssrc1: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x17E << 23) | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_sopk(op: int, sdst: int, simm16: int, literal: Optional[int] = None) -> bytes:
    w0 = (0xB << 28) | ((op & 0x1F) << 23) | ((sdst & 0x7F) << 16) | (simm16 & 0xFFFF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vop2(op: int, vdst: int, src0: int, vsrc1: int, literal: Optional[int] = None) -> bytes:
    w0 = ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vop1(op: int, vdst: int, src0: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x3F << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vopc(op: int, src0: int, vsrc1: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x3E << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vopcx(op: int, src0: int, vsrc1: int, literal: Optional[int] = None) -> bytes:
    return encode_vopc(op | 0x80, src0, vsrc1, literal)


def encode_vop3(
    op: int, vdst: int, src0: int, src1: int = 0, src2: int = 0, sdst: int = 0, literal: Optional[int] = None
) -> bytes:
    w0 = (0x35 << 26) | ((op & 0x3FF) << 16) | ((sdst & 0x7F) << 8) | (vdst & 0xFF)
    w1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return struct.pack("<III", w0, w1, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<II", w0, w1)


def encode_vop3p(
    op: int, vdst: int, src0: int, src1: int = 0, src2: int = 0, literal: Optional[int] = None
) -> bytes:
    w0 = (0xCC << 24) | ((op & 0x7F) << 16) | (vdst & 0xFF)
    w1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return struct.pack("<III", w0, w1, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<II", w0, w1)


def encode_vopd(
    opx: int, opy: int, vdstx: int, srcx0: int, vsrcx1: int, vdsty: int, srcy0: int, vsrcy1: int,
    literal: Optional[int] = None,
) -> bytes:
    if (vdstx & 1) == (vdsty & 1):
        raise ValueError(f"VOPD requires opposite VGPR dst banks: v{vdstx} and v{vdsty}")
    w0 = (0x32 << 26) | ((opx & 0xF) << 22) | ((opy & 0x1F) << 17) | ((vsrcx1 & 0xFF) << 9) | (srcx0 & 0x1FF)
    w1 = ((vdstx & 0xFF) << 24) | (((vdsty >> 1) & 0x7F) << 17) | ((vsrcy1 & 0xFF) << 9) | (srcy0 & 0x1FF)
    return struct.pack("<III", w0, w1, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<II", w0, w1)


def encode_ds(
    op: int, addr_v: int, data0_v: int = 0, data1_v: int = 0, vdst: int = 0, offset0: int = 0, offset1: int = 0
) -> bytes:
    w0 = (0x36 << 26) | ((op & 0xFF) << 18) | ((offset1 & 0xFF) << 8) | (offset0 & 0xFF)
    w1 = ((vdst & 0xFF) << 24) | ((data1_v & 0xFF) << 16) | ((data0_v & 0xFF) << 8) | (addr_v & 0xFF)
    return struct.pack("<II", w0, w1)


def encode_ds_offset16(op: int, addr_v: int, data0_v: int = 0, vdst: int = 0, offset16: int = 0) -> bytes:
    return encode_ds(op, addr_v, data0_v, 0, vdst, offset16 & 0xFF, (offset16 >> 8) & 0xFF)


def encode_smem(op: int, sdata: int, sbase: int, ioffset: int = 0, soffset: int = NULL_ENC) -> bytes:
    w0 = (0x3D << 26) | ((op & 0xFF) << 18) | ((sdata & 0x7F) << 6) | ((sbase >> 1) & 0x3F)
    w1 = ((soffset & 0x7F) << 25) | (ioffset & 0x1FFFFF)
    return struct.pack("<II", w0, w1)


def encode_global(op: int, vaddr: int, vdata: int = 0, vdst: int = 0, saddr: int = NULL_ENC, offset: int = 0) -> bytes:
    w0 = (0x37 << 26) | ((op & 0x7F) << 18) | (2 << 16) | (offset & 0x1FFF)
    w1 = ((vdst & 0xFF) << 24) | ((saddr & 0x7F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)
    return struct.pack("<II", w0, w1)


def encode_mubuf(
    op: int, vdata: int, vaddr: int = 0, srsrc: int = 0, soffset: int = 0x80,
    offset: int = 0, offen: int = 0, idxen: int = 0,
) -> bytes:
    w0 = (0x38 << 26) | ((op & 0xFF) << 18) | (offset & 0xFFF)
    w1 = (
        ((soffset & 0xFF) << 24)
        | ((idxen & 1) << 23)
        | ((offen & 1) << 22)
        | (((srsrc >> 2) & 0x1F) << 16)
        | ((vdata & 0xFF) << 8)
        | (vaddr & 0xFF)
    )
    return struct.pack("<II", w0, w1)

# ---------------------------------------------------------------------------
# Genuine AMD GFX11 (RDNA3 Layout 3 / RDNA4 Layout 4) Hardware SQTT v2 Codec
# ---------------------------------------------------------------------------

class SQTTPacketKind(str, Enum):
    """Official AMD GFX11 (RDNA3 / RDNA4) SQTT v2 hardware packet types."""

    NOP = "NOP"
    LAYOUT_HEADER = "LAYOUT_HEADER"
    WAVESTART = "WAVESTART"
    WAVEEND = "WAVEEND"
    WAVEALLOC = "WAVEALLOC"
    WAVERDY = "WAVERDY"
    VALUINST = "VALUINST"
    INST = "INST"
    IMMEDIATE = "IMMEDIATE"
    IMMEDIATE_MASK = "IMMEDIATE_MASK"
    ALUEXEC = "ALUEXEC"
    VMEMEXEC = "VMEMEXEC"
    TS_DELTA_SHORT = "TS_DELTA_SHORT"
    TS_DELTA_OR_MARK = "TS_DELTA_OR_MARK"
    TS_WAVE_STATE = "TS_WAVE_STATE"
    TS_DELTA_S5_W2 = "TS_DELTA_S5_W2"
    TS_DELTA_S5_W3 = "TS_DELTA_S5_W3"
    TS_DELTA_S8_W3 = "TS_DELTA_S8_W3"
    REG = "REG"
    SNAPSHOT = "SNAPSHOT"
    EVENT = "EVENT"
    EVENT_BIG = "EVENT_BIG"
    PERF = "PERF"
    UTILCTR = "UTILCTR"


class SQTTMemSrc(Enum):
    """VMEMEXEC packet `src` field."""

    LDS = 0
    LDS_ALT = 1
    VMEM = 2
    VMEM_ALT = 3


class SQTTAluSrc(Enum):
    """ALUEXEC packet `src` field."""

    NONE = 0
    SALU = 1
    VALU = 2
    VALU_SALU = 3


# Official AMD RDNA3 (GFX1100) SQTT INST packet 7-bit opcodes
SQTT_INST_OP_NAMES: Dict[int, str] = {
    0x00: "SALU",
    0x01: "SMEM_RD",
    0x03: "JUMP",
    0x04: "JUMP_NO",
    0x05: "CALL",
    0x09: "MESSAGE",
    0x0B: "VALUT_4",
    0x0D: "VALUB_2",
    0x0E: "VALUB_4",
    0x0F: "VALUB_16",
    0x12: "VINTERP",
    0x13: "BARRIER",
    0x1C: "FLAT_RD_2",
    0x1D: "FLAT_WR_3",
    0x1E: "FLAT_WR_4",
    0x1F: "FLAT_WR_5",
    0x20: "FLAT_WR_6",
    0x21: "SGMEM_RD_1",
    0x22: "SGMEM_RD_2",
    0x24: "SGMEM_WR_2",
    0x25: "SGMEM_WR_3",
    0x26: "SGMEM_WR_4",
    0x27: "SGMEM_WR_5",
    0x28: "SGMEM_WR_6",
    0x29: "LDS_RD",
    0x2A: "LDS_WR_1",
    0x2B: "LDS_WR_2",
    0x2C: "LDS_WR_3",
    0x2D: "LDS_WR_4",
    0x2E: "LDS_WR_5",
    0x72: "SALU_WR_EXEC",
    0x73: "VALU1_WR_EXEC",
}
for _cat, _s, _e, _bc in (("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0x5A, 0x66, 1)):
    for _v in range(_s, _e + 1):
        SQTT_INST_OP_NAMES[_v] = f"OTHER_{_cat}_{_v - _s + _bc}"

SQTT_INST_OP_BY_NAME: Dict[str, int] = {v: k for k, v in SQTT_INST_OP_NAMES.items()}

SQTT_RDNA4_INST_OP_NAMES: Dict[int, str] = {
    0x00: "SALU", 0x01: "SMEM", 0x02: "SMEM_WR", 0x03: "JUMP", 0x04: "JUMP_NO", 0x05: "CALL",
    0x07: "SALU_NO_EXEC", 0x09: "MESSAGE", 0x0A: "VALU_1", 0x0B: "VALUT_4", 0x0C: "VALUB_1",
    0x0D: "VALUB_2", 0x0E: "VALUB_4", 0x0F: "VALUB_16", 0x12: "VINTERP", 0x13: "BARRIER_WAIT",
    0x1C: "FLAT_RD_2", 0x1D: "FLAT_WR_3", 0x1E: "FLAT_WR_4", 0x1F: "FLAT_WR_5", 0x20: "FLAT_WR_6",
    0x21: "VMEM_RD_1", 0x22: "VMEM_RD_2", 0x23: "VMEM_WR_1", 0x24: "VMEM_WR_2", 0x25: "VMEM_WR_3",
    0x26: "VMEM_WR_4", 0x27: "VMEM_WR_5", 0x28: "VMEM_WR_6",
    0x29: "LDS_RD", 0x2A: "LDS_WR_1", 0x2B: "LDS_WR_2", 0x2C: "LDS_WR_3", 0x2D: "LDS_WR_4", 0x2E: "LDS_WR_5",
    0x72: "SALU_WR_EXEC", 0x73: "VALU1_WR_EXEC",
}
for _cat, _s, _e, _bc in (("LDS", 0x50, 0x54, 1), ("FLAT", 0x55, 0x59, 2), ("VMEM", 0x5A, 0x66, 1)):
    for _v in range(_s, _e + 1):
        SQTT_RDNA4_INST_OP_NAMES[_v] = f"OTHER_{_cat}_{_v - _s + _bc}"


@dataclass(frozen=True)
class SQTTPacket:
    """Decoded AMD GFX11 (RDNA3 / RDNA4) hardware SQTT v2 packet."""

    kind: SQTTPacketKind
    timestamp: int
    delta: int = 0
    wave_id: int = 0
    simd_id: int = 0
    cu_id: int = 0
    op_code: int = 0
    op_name: str = ""
    flag1: int = 0
    flag2: int = 0
    mask: int = 0
    src_name: str = ""
    layout: int = 3
    raw: int = 0
    nibble_len: int = 1


@dataclass(frozen=True)
class SQTTAlignedEvent:
    """Real hardware SQTT packet aligned to a decoded GFX11 `.text` instruction."""

    wave_id: int
    simd_id: int
    cu_id: int
    pc: int
    timestamp: int
    issue_time: int
    dt_prev: int
    stall: int
    duration: int
    packet_kind: SQTTPacketKind
    op_name: str
    insn: DecodedInsn

    def to_sqtt_tuple(self) -> Tuple[int, int, int, int]:
        return (self.pc, self.stall, self.duration, self.issue_time)


# Hardware packet spec table: (kind, enc_mask, enc_val, nibbles, delta_lo, delta_mask, special)
# special: 0=normal, 1=TS_DELTA_OR_MARK (RDNA3), 2=TS_DELTA_SHORT (+4), 3=TS_DELTA_OR_MARK (RDNA4)
_RDNA3_PKT_SPECS: List[Tuple[SQTTPacketKind, int, int, int, int, int, int]] = [
    (SQTTPacketKind.EVENT,            0xFF, 0b01100001,  6,  8, 0x7, 0),
    (SQTTPacketKind.EVENT_BIG,        0xFF, 0b11100001,  8,  8, 0x7, 0),
    (SQTTPacketKind.TS_DELTA_S8_W3,   0x7F, 0b0100001,  18,  8, 0x7, 0),
    (SQTTPacketKind.TS_DELTA_OR_MARK, 0x7F, 0b0000001,  12, 12, (1 << 36) - 1, 1),
    (SQTTPacketKind.TS_WAVE_STATE,    0x7F, 0b1010001,   6,  7, 0x1FF, 0),
    (SQTTPacketKind.SNAPSHOT,         0x7F, 0b1110001,  16,  7, 0x7, 0),
    (SQTTPacketKind.LAYOUT_HEADER,    0x7F, 0b0010001,  16,  0, 0, 0),
    (SQTTPacketKind.UTILCTR,          0x7F, 0b0110001,  12,  7, 0x3, 0),
    (SQTTPacketKind.TS_DELTA_S5_W3,   0x1F, 0b00110,    13,  5, 0x7, 0),
    (SQTTPacketKind.TS_DELTA_S5_W2,   0x1F, 0b11100,    12,  5, 0x3, 0),
    (SQTTPacketKind.IMMEDIATE_MASK,   0x1F, 0b00100,     6,  5, 0x7, 0),
    (SQTTPacketKind.WAVERDY,          0x1F, 0b10100,     6,  5, 0x7, 0),
    (SQTTPacketKind.WAVEEND,          0x1F, 0b10101,     5,  5, 0x7, 0),
    (SQTTPacketKind.WAVESTART,        0x1F, 0b01100,     8,  5, 0x3, 0),
    (SQTTPacketKind.WAVEALLOC,        0x1F, 0b00101,     5,  5, 0x7, 0),
    (SQTTPacketKind.PERF,             0x1F, 0b10110,     7,  5, 0x7, 0),
    (SQTTPacketKind.VMEMEXEC,         0x0F, 0b1111,      2,  4, 0x3, 0),
    (SQTTPacketKind.ALUEXEC,          0x0F, 0b1110,      2,  4, 0x3, 0),
    (SQTTPacketKind.IMMEDIATE,        0x0F, 0b1101,      3,  4, 0x7, 0),
    (SQTTPacketKind.TS_DELTA_SHORT,   0x0F, 0b1000,      2,  4, 0xF, 2),
    (SQTTPacketKind.REG,              0x0F, 0b1001,     16,  4, 0x7, 0),
    (SQTTPacketKind.NOP,              0x0F, 0b0000,      1,  0, 0, 0),
    (SQTTPacketKind.VALUINST,         0x07, 0b011,       3,  3, 0x7, 0),
    (SQTTPacketKind.INST,             0x07, 0b010,       5,  4, 0x7, 0),
]

_RDNA4_OVERRIDES: Dict[SQTTPacketKind, Tuple[int, int, int, int]] = {
    # kind -> (nibbles, delta_lo, delta_mask, special)
    SQTTPacketKind.TS_DELTA_S5_W3: (14, 7, 0x7, 0),
    SQTTPacketKind.TS_DELTA_S5_W2: (10, 5, 0x3, 0),
    SQTTPacketKind.WAVEALLOC: (6, 5, 0x7, 0),
    SQTTPacketKind.PERF: (8, 7, 0x7, 0),
    SQTTPacketKind.TS_DELTA_OR_MARK: (16, 12, (1 << 52) - 1, 3),
    SQTTPacketKind.INST: (5, 3, 0x7, 0),
}


def _build_sqtt_state_table(
    specs: Sequence[Tuple[SQTTPacketKind, int, int, int, int, int, int]]
) -> Tuple[List[Tuple[SQTTPacketKind, int, int, int, int, int, int]], bytes]:
    ordered = sorted(specs, key=lambda x: (-bin(x[1]).count("1"), x[0] is SQTTPacketKind.NOP))
    nop_idx = next(i for i, s in enumerate(ordered) if s[0] is SQTTPacketKind.NOP)
    tbl = bytes(
        next((i for i, s in enumerate(ordered) if (b & s[1]) == s[2]), nop_idx)
        for b in range(256)
    )
    return ordered, tbl


_RDNA3_SPECS_ORDERED, _RDNA3_STATE_TABLE = _build_sqtt_state_table(_RDNA3_PKT_SPECS)
_RDNA4_PKT_SPECS = [
    (k, m, v, *_RDNA4_OVERRIDES[k]) if k in _RDNA4_OVERRIDES else (k, m, v, nib, dlo, dmk, sp)
    for (k, m, v, nib, dlo, dmk, sp) in _RDNA3_PKT_SPECS
]
_RDNA4_SPECS_ORDERED, _RDNA4_STATE_TABLE = _build_sqtt_state_table(_RDNA4_PKT_SPECS)


class _SQTTNibbleWriter:
    """Little-endian nibble packer for genuine AMD GFX11 SQTT v2 hardware streams."""

    def __init__(self) -> None:
        self._nibbles: List[int] = []

    def emit(self, raw_bits: int, nibbles: int) -> None:
        for i in range(nibbles):
            self._nibbles.append((raw_bits >> (4 * i)) & 0xF)

    def to_bytes(self, pad_lookahead: bool = True) -> bytes:
        # Append 16 NOP nibbles (8 zero bytes) so the 64-bit sliding hardware lookahead
        # window flushes every emitted packet through bit 0 during decoding.
        nibs = list(self._nibbles)
        if len(nibs) % 2 == 1:
            nibs.append(0)
        if pad_lookahead:
            nibs.extend([0] * 16)
        out = bytearray(len(nibs) // 2)
        for i in range(0, len(nibs), 2):
            out[i // 2] = nibs[i] | (nibs[i + 1] << 4)
        return bytes(out)


def parse_sqtt(
    stream: Union[bytes, bytearray, memoryview, str],
    include_nops: bool = False,
) -> List[SQTTPacket]:
    """Decode real AMD GFX11 (RDNA3 Layout 3) or RDNA4 (Layout 4) hardware SQTT v2 bytes.

    Uses the exact ROCprofv3 / AMDGPU KFD 64-bit sliding nibble state machine.
    """
    data = _normalize_bytes(stream)
    n = len(data)
    if n < 8:
        return []

    specs = _RDNA3_SPECS_ORDERED
    state_tbl = _RDNA3_STATE_TABLE
    is_rdna4 = False
    layout = 3
    reg = 0
    pos = 0
    nib_off = 0
    nib_count = 16
    time = 0
    packets: List[SQTTPacket] = []

    while pos + ((nib_count + nib_off + 1) >> 1) <= n:
        need = nib_count - nib_off
        if nib_off:
            reg = (reg >> 4) | ((data[pos] >> 4) << 60)
            pos += 1
        byte_count = need >> 1
        if byte_count:
            rb = min(byte_count, 8)
            chunk = int.from_bytes(data[pos:pos + rb], "little")
            reg = (reg >> (rb * 8)) | (chunk << (64 - rb * 8))
            pos += byte_count
        nib_off = need & 1
        if nib_off:
            reg = (reg >> 4) | ((data[pos] & 0xF) << 60)

        idx = state_tbl[reg & 0xFF]
        kind, _, _, nib_count, d_lo, d_mask, spec = specs[idx]
        delta = (reg >> d_lo) & d_mask if d_mask else 0
        if spec == 1:  # RDNA3 TS_DELTA_OR_MARK
            pl = (reg >> 8) & 1
            rt = (reg >> 9) & 1
            if rt and not pl:
                delta = 0
        elif spec == 3:  # RDNA4 TS_DELTA_OR_MARK
            rt = (reg >> 7) & 1
            pl = (reg >> 8) & 1
            if rt and not pl:
                delta = 0
        elif spec == 2:  # TS_DELTA_SHORT
            delta += 4

        time += delta

        if kind is SQTTPacketKind.NOP:
            if include_nops:
                packets.append(SQTTPacket(kind=kind, timestamp=time, delta=0, raw=reg & 0xF, nibble_len=1))
            continue

        wave_id = 0
        simd_id = 0
        cu_id = 0
        op_code = 0
        op_name = ""
        flag1 = 0
        flag2 = 0
        mask = 0
        src_name = ""

        if kind is SQTTPacketKind.LAYOUT_HEADER:
            layout = (reg >> 7) & 0x3F
            simd_id = (reg >> 13) & 0x3
            if layout == 4 and not is_rdna4:
                is_rdna4 = True
                specs = _RDNA4_SPECS_ORDERED
                state_tbl = _RDNA4_STATE_TABLE
        elif kind is SQTTPacketKind.WAVESTART:
            sa = (reg >> 7) & 1
            simd_id = (reg >> 8) & 0x3
            if is_rdna4:
                wgp = (reg >> 10) & 0xF
                wave_id = (reg >> 15) & 0x1F
                cu_id = wgp | (sa << 4)
            else:
                wgp = (reg >> 10) & 0x7
                wave_id = (reg >> 13) & 0x1F
                cu_id = wgp | (sa << 3)
        elif kind is SQTTPacketKind.WAVEEND:
            sa = (reg >> 8) & 1
            simd_id = (reg >> 9) & 0x3
            if is_rdna4:
                wgp = (reg >> 11) & 0xF
                wave_id = (reg >> 15) & 0x1F
                cu_id = wgp | (sa << 4)
            else:
                wgp = (reg >> 11) & 0x7
                wave_id = (reg >> 15) & 0x1F
                cu_id = wgp | (sa << 3)
        elif kind is SQTTPacketKind.VALUINST:
            flag1 = (reg >> 6) & 1
            wave_id = (reg >> 7) & 0x1F
            op_name = "VALUINST"
        elif kind is SQTTPacketKind.INST:
            if is_rdna4:
                flag1 = (reg >> 6) & 1
                wave_id = (reg >> 7) & 0x1F
                op_code = (reg >> 12) & 0xFF
                op_name = SQTT_RDNA4_INST_OP_NAMES.get(op_code, f"INST_0x{op_code:02x}")
            else:
                flag1 = (reg >> 3) & 1
                flag2 = (reg >> 7) & 1
                wave_id = (reg >> 8) & 0x1F
                op_code = (reg >> 13) & 0x7F
                op_name = SQTT_INST_OP_NAMES.get(op_code, f"INST_0x{op_code:02x}")
        elif kind is SQTTPacketKind.IMMEDIATE:
            wave_id = (reg >> 7) & 0x1F
            op_name = "IMMEDIATE"
        elif kind in (SQTTPacketKind.IMMEDIATE_MASK, SQTTPacketKind.WAVERDY):
            mask = (reg >> 8) & 0xFFFF
        elif kind is SQTTPacketKind.ALUEXEC:
            src_name = SQTTAluSrc((reg >> 6) & 0x3).name
        elif kind is SQTTPacketKind.VMEMEXEC:
            src_name = SQTTMemSrc((reg >> 6) & 0x3).name

        pkt_mask = (1 << min(64, nib_count * 4)) - 1
        packets.append(SQTTPacket(
            kind=kind, timestamp=time, delta=delta,
            wave_id=wave_id, simd_id=simd_id, cu_id=cu_id,
            op_code=op_code, op_name=op_name, flag1=flag1, flag2=flag2,
            mask=mask, src_name=src_name, layout=layout,
            raw=reg & pkt_mask, nibble_len=nib_count,
        ))

    return packets


def _emit_sqtt_delta(w: _SQTTNibbleWriter, delta: int) -> int:
    """Advance hardware SQTT timestamp by emitting TS_DELTA_OR_MARK / TS_DELTA_SHORT; returns remainder (0..3)."""
    if delta <= 0:
        return 0
    while delta >= 20:
        chunk = min(delta, (1 << 36) - 1)
        # TS_DELTA_OR_MARK: enc[6:0]=0b0000001, pl=0, rt=0, delta=[47:12], 12 nibbles
        raw = 0b0000001 | (chunk << 12)
        w.emit(raw, 12)
        delta -= chunk
    if delta >= 4:
        # TS_DELTA_SHORT: enc[3:0]=0b1000, delta_field = delta - 4 in [7:4], 2 nibbles
        raw = 0b1000 | (((delta - 4) & 0xF) << 4)
        w.emit(raw, 2)
        return 0
    return delta


def _pipe_to_sqtt_inst_op(ev: CycleEvent) -> Tuple[str, int]:
    """Map a simulated CycleEvent to its real AMD GFX11 SQTT v2 packet kind + op_code."""
    p = ev.pipe
    if p is Pipe.VALU32:
        return ("VALUINST", 0)
    if p is Pipe.WAIT_LGKMCNT or ev.mnemonic in ("s_nop", "s_clause"):
        return ("IMMEDIATE", 0)
    if p is Pipe.SALU:
        return ("INST", SQTT_INST_OP_BY_NAME["SALU"])
    if p is Pipe.SALU_WR_EXEC:
        return ("INST", SQTT_INST_OP_BY_NAME["SALU_WR_EXEC"])
    if p is Pipe.VALU1_WR_EXEC:
        return ("INST", SQTT_INST_OP_BY_NAME["VALU1_WR_EXEC"])
    if p is Pipe.JUMP:
        return ("INST", SQTT_INST_OP_BY_NAME["JUMP" if ev.taken else "JUMP_NO"])
    if p is Pipe.BARRIER:
        return ("INST", SQTT_INST_OP_BY_NAME["BARRIER"])
    if p is Pipe.VALUT_4:
        return ("INST", SQTT_INST_OP_BY_NAME["VALUT_4"])
    if p is Pipe.VALUB_2:
        return ("INST", SQTT_INST_OP_BY_NAME["VALUB_2"])
    if p is Pipe.VALUB_4:
        return ("INST", SQTT_INST_OP_BY_NAME["VALUB_4"])
    if p in (Pipe.VALUB_16, Pipe.VALU_WMMA):
        return ("INST", SQTT_INST_OP_BY_NAME["VALUB_16"])
    if p is Pipe.VINTERP:
        return ("INST", SQTT_INST_OP_BY_NAME["VINTERP"])
    if p is Pipe.SMEM_RD:
        return ("INST", SQTT_INST_OP_BY_NAME["SMEM_RD"])
    if p is Pipe.LDS_RD:
        return ("INST", SQTT_INST_OP_BY_NAME["LDS_RD"])
    if p in LDS_WR_PIPES:
        return ("INST", SQTT_INST_OP_BY_NAME[p.value])
    if p is Pipe.VMEM_RD:
        return ("INST", SQTT_INST_OP_BY_NAME["SGMEM_RD_1"])
    if p is Pipe.VMEM_WR:
        return ("INST", SQTT_INST_OP_BY_NAME["SGMEM_WR_2"])
    return ("IMMEDIATE", 0)


def encode_sqtt(
    events: Union[Sequence[CycleEvent], Mapping[int, Sequence[CycleEvent]]],
    base_timestamp: int = 0,
    cu_id: int = 0,
    simd_id: Optional[int] = None,
    emit_endpgm_pkt: bool = False,
) -> bytes:
    """Encode simulated CycleEvent trace(s) into genuine AMD GFX11 (RDNA3 Layout 3) SQTT v2 bytes.

    Produces standards-compliant ROCprofv3 / AMDGPU KFD hardware nibble packets
    (`LAYOUT_HEADER`, `WAVESTART`, `TS_DELTA_OR_MARK`, `TS_DELTA_SHORT`, `VALUINST`,
    `INST`, `IMMEDIATE`, `ALUEXEC`, `VMEMEXEC`, `WAVEEND`) — zero custom/synthetic headers.
    """
    if isinstance(events, Mapping):
        flat: List[CycleEvent] = []
        for evs in events.values():
            flat.extend(evs)
        flat.sort(key=lambda e: (e.issue_cycle, e.wave_id, e.pc))
    else:
        flat = list(events)

    trace_simd = simd_id if simd_id is not None else (flat[0].simd_id if flat else 0)
    w = _SQTTNibbleWriter()

    # 1. LAYOUT_HEADER: enc[6:0]=0b0010001, layout[12:7]=3, simd[14:13]=trace_simd, 16 nibbles
    hdr_raw = 0b0010001 | (3 << 7) | ((trace_simd & 0x3) << 13)
    w.emit(hdr_raw, 16)

    cur_time = 0
    if base_timestamp > 0:
        rem = _emit_sqtt_delta(w, base_timestamp)
        if rem:
            # Emit exact remainder via TS_DELTA_OR_MARK
            w.emit(0b0000001 | (rem << 12), 12)
        cur_time = base_timestamp

    started_waves: Set[int] = set()
    wave_simd: Dict[int, int] = {}
    cu_lo = cu_id & 0x7
    sa_bit = (cu_id >> 3) & 1

    for ev in flat:
        if ev.pipe is Pipe.DELAY_ALU:
            continue
        if ev.pipe is Pipe.ENDPGM and not emit_endpgm_pkt:
            continue

        ev_time = base_timestamp + ev.issue_cycle
        delta = max(0, ev_time - cur_time)
        if ev.wave_id not in started_waves:
            started_waves.add(ev.wave_id)
            wave_simd[ev.wave_id] = ev.simd_id
            rem = _emit_sqtt_delta(w, delta)
            if rem > 3:
                w.emit(0b0000001 | (rem << 12), 12)
                rem = 0
            # WAVESTART: enc[4:0]=0b01100, delta[6:5], sa[7], simd[9:8], wgp[12:10], wave[17:13]
            ws_raw = (
                0b01100
                | ((rem & 0x3) << 5)
                | (sa_bit << 7)
                | ((ev.simd_id & 0x3) << 8)
                | (cu_lo << 10)
                | ((ev.wave_id & 0x1F) << 13)
            )
            w.emit(ws_raw, 8)
            cur_time = ev_time
            delta = 0

        if delta > 7:
            rem = _emit_sqtt_delta(w, delta)
        else:
            rem = delta
        cur_time = ev_time

        pkind, op_code = _pipe_to_sqtt_inst_op(ev)
        if pkind == "VALUINST":
            # VALUINST: enc[2:0]=0b011, delta[5:3], flag[6], wave[11:7], 3 nibbles
            raw = 0b011 | ((rem & 0x7) << 3) | ((ev.wave_id & 0x1F) << 7)
            w.emit(raw, 3)
        elif pkind == "IMMEDIATE":
            # IMMEDIATE: enc[3:0]=0b1101, delta[6:4], wave[11:7], 3 nibbles
            raw = 0b1101 | ((rem & 0x7) << 4) | ((ev.wave_id & 0x1F) << 7)
            w.emit(raw, 3)
        else:
            # INST: enc[2:0]=0b010, flag1[3]=0, delta[6:4], flag2[7]=0, wave[12:8], op[19:13], 5 nibbles
            raw = 0b010 | ((rem & 0x7) << 4) | ((ev.wave_id & 0x1F) << 8) | ((op_code & 0x7F) << 13)
            w.emit(raw, 5)

        # Emit hardware ALUEXEC / VMEMEXEC sideband completion marker (delta=0)
        if ev.alu_src in (AluSrc.VALU, AluSrc.SALU, AluSrc.VALU_SALU, AluSrc.VOPD):
            asrc = (
                SQTTAluSrc.VALU_SALU.value if ev.alu_src is AluSrc.VALU_SALU
                else (SQTTAluSrc.SALU.value if ev.alu_src is AluSrc.SALU else SQTTAluSrc.VALU.value)
            )
            w.emit(0b1110 | (asrc << 6), 2)
        elif ev.pipe in LDS_PIPES:
            w.emit(0b1111 | (SQTTMemSrc.LDS.value << 6), 2)
        elif ev.pipe in VMEM_PIPES:
            w.emit(0b1111 | (SQTTMemSrc.VMEM.value << 6), 2)

    for wid in sorted(started_waves):
        # WAVEEND: enc[4:0]=0b10101, delta[7:5]=1, sa[8], simd[10:9], wgp[13:11], wave[19:15], 5 nibbles
        we_raw = (
            0b10101
            | (1 << 5)
            | (sa_bit << 8)
            | ((wave_simd[wid] & 0x3) << 9)
            | (cu_lo << 11)
            | ((wid & 0x1F) << 15)
        )
        w.emit(we_raw, 5)

    return w.to_bytes(pad_lookahead=True)


def extract_sqtt_contention(
    data: Union[bytes, bytearray, memoryview, str, Sequence[SQTTPacket]],
    base_timestamp: Optional[int] = None,
) -> List[ContentionEvent]:
    """Extract cross-SIMD CU LDS/VMEM contention events from real hardware SQTT v2 packets."""
    pkts = parse_sqtt(data) if isinstance(data, (bytes, bytearray, memoryview, str)) else list(data)
    if base_timestamp is None:
        inst_ts = [p.timestamp for p in pkts if p.kind in (SQTTPacketKind.VALUINST, SQTTPacketKind.INST, SQTTPacketKind.IMMEDIATE)]
        base_timestamp = min(inst_ts) if inst_ts else 0

    out: List[ContentionEvent] = []
    for p in pkts:
        if p.kind is SQTTPacketKind.INST and p.op_name.startswith("OTHER_"):
            parts = p.op_name.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                dur = max(1, int(parts[2]))
                port = ContentionPort.OTHER_LDS if parts[1] == "LDS" else ContentionPort.OTHER_VMEM
                start = max(0, p.timestamp - base_timestamp)
                out.append(ContentionEvent(port=port, start_cycle=start, duration_cycles=dur))
    return out


_SQTT_NON_EMITTING_MNEMONICS: frozenset[str] = frozenset({
    "s_delay_alu", "s_wait_alu", "s_barrier_wait", "s_code_end",
})


def align_sqtt_to_program(
    sqtt_data: Union[bytes, bytearray, memoryview, str, Sequence[SQTTPacket]],
    program: Union[bytes, bytearray, memoryview, str, Sequence[DecodedInsn], AMDGPUKernel],
    *,
    cu: Optional[int] = None,
    base_pc: Optional[int] = None,
    normalize_to_first_inst: bool = True,
) -> Dict[Tuple[int, int], List[SQTTAlignedEvent]]:
    """Align real AMD GFX11 SQTT v2 packets to decoded ELF `.text` / raw GFX11 instructions.

    Mirrors ROCprofv3 / AMDGPU KFD instruction-PC reconstruction:
      - Follows `LAYOUT_HEADER.simd` to identify the traced SIMD.
      - Initializes `wave_pc[(simd, wave)] = entry_pc` at `WAVESTART`.
      - Advances PC on `VALUINST`, `IMMEDIATE`, `IMMEDIATE_MASK`, and traced-SIMD `INST`
        (taking `branch_target` on `INST(JUMP)` and fallthrough on `INST(JUMP_NO)`),
        skipping non-SQTT scheduler hints (`s_delay_alu`, `s_wait_alu`, `s_code_end`).
      - Returns `{(simd_id, wave_id): [SQTTAlignedEvent, ...]}` with exact hardware
        timestamps, inter-instruction deltas, and aligned `DecodedInsn` records.
    """
    pkts = parse_sqtt(sqtt_data) if isinstance(sqtt_data, (bytes, bytearray, memoryview, str)) else list(sqtt_data)
    if isinstance(program, AMDGPUKernel):
        decoded = decode_gfx11(program.text_bytes, base_pc=program.text_addr if base_pc is None else base_pc, stop_at_code_end=True)
    elif isinstance(program, (bytes, bytearray, memoryview, str)):
        decoded = decode_gfx11(program, base_pc=0 if base_pc is None else base_pc)
    else:
        decoded = list(program)
    if not decoded:
        return {}

    pc_map: Dict[int, DecodedInsn] = {d.pc: d for d in decoded}
    entry_pc = decoded[0].pc

    # Auto-detect target CU if caller didn't force one: prefer CU=0 when present, else first WAVESTART CU
    target_cu = cu
    if target_cu is None:
        ws_cus = [p.cu_id for p in pkts if p.kind is SQTTPacketKind.WAVESTART]
        target_cu = 0 if 0 in ws_cus else (ws_cus[0] if ws_cus else 0)

    traced_simd = 0
    wave_pc: Dict[Tuple[int, int], int] = {}
    wave_start_ts: Dict[Tuple[int, int], int] = {}
    raw_by_wave: Dict[Tuple[int, int], List[Tuple[int, SQTTPacketKind, str, DecodedInsn]]] = {}

    def next_emitting(key: Tuple[int, int]) -> Optional[DecodedInsn]:
        pc = wave_pc.get(key)
        while pc is not None and pc in pc_map and pc_map[pc].mnemonic in _SQTT_NON_EMITTING_MNEMONICS:
            pc += pc_map[pc].size
            wave_pc[key] = pc
        if pc is None or pc not in pc_map:
            return None
        return pc_map[pc]

    for p in pkts:
        if p.kind is SQTTPacketKind.LAYOUT_HEADER:
            traced_simd = p.simd_id
        elif p.kind is SQTTPacketKind.WAVESTART:
            if p.cu_id == target_cu:
                key = (p.simd_id, p.wave_id)
                wave_pc[key] = entry_pc
                wave_start_ts.setdefault(key, p.timestamp)
                raw_by_wave.setdefault(key, [])
        elif p.kind is SQTTPacketKind.WAVEEND:
            if p.cu_id == target_cu:
                wave_pc.pop((p.simd_id, p.wave_id), None)
        elif p.kind is SQTTPacketKind.IMMEDIATE_MASK:
            for wid in range(16):
                if p.mask & (1 << wid):
                    key = (traced_simd, wid)
                    insn = next_emitting(key)
                    if insn is not None:
                        wave_pc[key] = insn.pc + insn.size
                        raw_by_wave.setdefault(key, []).append((p.timestamp, p.kind, "IMMEDIATE_MASK", insn))
        elif p.kind in (SQTTPacketKind.VALUINST, SQTTPacketKind.IMMEDIATE, SQTTPacketKind.INST):
            if p.kind is SQTTPacketKind.INST and p.op_name.startswith("OTHER_"):
                continue
            key = (traced_simd, p.wave_id)
            if key not in wave_pc:
                # Stream lacked explicit WAVESTART for this wave
                wave_pc[key] = entry_pc
                wave_start_ts.setdefault(key, p.timestamp)
            insn = next_emitting(key)
            if insn is None:
                continue
            if p.kind is SQTTPacketKind.INST and p.op_name == "JUMP" and insn.branch_target is not None:
                wave_pc[key] = insn.branch_target
            else:
                wave_pc[key] = insn.pc + insn.size
            raw_by_wave.setdefault(key, []).append((p.timestamp, p.kind, p.op_name or p.kind.value, insn))

    aligned: Dict[Tuple[int, int], List[SQTTAlignedEvent]] = {}
    for key, entries in raw_by_wave.items():
        if not entries:
            continue
        t0 = entries[0][0] if normalize_to_first_inst else wave_start_ts.get(key, entries[0][0])
        evs: List[SQTTAlignedEvent] = []
        prev_t = entries[0][0]
        for idx, (ts, pkind, op_name, insn) in enumerate(entries):
            dt = 0 if idx == 0 else max(0, ts - prev_t)
            next_dt = max(1, entries[idx + 1][0] - ts) if idx + 1 < len(entries) else 1
            stall = max(0, dt - 1) if idx > 0 else 0
            evs.append(SQTTAlignedEvent(
                wave_id=key[1], simd_id=key[0], cu_id=target_cu,
                pc=insn.pc, timestamp=ts, issue_time=ts - t0,
                dt_prev=dt, stall=stall, duration=next_dt,
                packet_kind=pkind, op_name=op_name, insn=insn,
            ))
            prev_t = ts
        aligned[key] = evs
    return aligned


_SQTT_INST_KINDS: frozenset[SQTTPacketKind] = frozenset({
    SQTTPacketKind.VALUINST,
    SQTTPacketKind.INST,
    SQTTPacketKind.IMMEDIATE,
    SQTTPacketKind.IMMEDIATE_MASK,
})


def sqtt_to_tuples(
    data: Union[bytes, bytearray, memoryview, str, Sequence[SQTTPacket]],
    program: Optional[Union[bytes, bytearray, memoryview, str, Sequence[DecodedInsn], AMDGPUKernel]] = None,
    normalize_base_time: bool = True,
    cu: Optional[int] = None,
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Extract per-wave SQTT `(pc, stall, dur, time)` tuples from real GFX11 SQTT v2 data.

    When `program` (raw GFX11 bytes, AMDGPU ELF, or DecodedInsn list) is supplied,
    aligns packets to exact `.text` PCs via `align_sqtt_to_program()`.
    """
    if program is not None:
        aligned = align_sqtt_to_program(data, program, cu=cu, normalize_to_first_inst=normalize_base_time)
        by_wave: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for (_simd, wid), evs in sorted(aligned.items()):
            by_wave[wid] = [e.to_sqtt_tuple() for e in evs]
        return by_wave

    pkts = parse_sqtt(data) if isinstance(data, (bytes, bytearray, memoryview, str)) else list(data)
    raw_by_wave: Dict[int, List[Tuple[int, SQTTPacket]]] = {}
    for p in pkts:
        if p.kind is SQTTPacketKind.IMMEDIATE_MASK:
            for wid in range(16):
                if p.mask & (1 << wid):
                    raw_by_wave.setdefault(wid, []).append((p.timestamp, p))
        elif p.kind in (SQTTPacketKind.VALUINST, SQTTPacketKind.IMMEDIATE):
            raw_by_wave.setdefault(p.wave_id, []).append((p.timestamp, p))
        elif p.kind is SQTTPacketKind.INST and not p.op_name.startswith("OTHER_"):
            raw_by_wave.setdefault(p.wave_id, []).append((p.timestamp, p))

    out: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for wid, items in raw_by_wave.items():
        if not items:
            continue
        t0 = items[0][0] if normalize_base_time else 0
        tuples: List[Tuple[int, int, int, int]] = []
        prev_t = items[0][0]
        for idx, (ts, _pkt) in enumerate(items):
            dt = 0 if idx == 0 else max(0, ts - prev_t)
            dur = max(1, items[idx + 1][0] - ts) if idx + 1 < len(items) else 1
            stall = max(0, dt - 1) if idx > 0 else 0
            tuples.append((idx * 4, stall, dur, ts - t0))
            prev_t = ts
        out[wid] = tuples
    return out

# ---------------------------------------------------------------------------
# WaveSpec, TraceSample & 32-Bank Shared SRAM (LDS) Conflict Engine
# ---------------------------------------------------------------------------

ProgramInput = Union[bytes, bytearray, memoryview, str, AMDGPUKernel, Sequence[DecodedInsn], Sequence[Instruction]]
ObservedInput = Union[
    bytes, bytearray, memoryview, str,
    Sequence[SQTTPacket], Sequence[SQTTAlignedEvent], Sequence[CycleEvent], Sequence[Tuple[int, int, int, int]]
]


@dataclass(frozen=True)
class WaveSpec:
    """Declarative wave launch descriptor for `simulate_cu(waves)`."""

    program: ProgramInput
    wave_id: int = 0
    simd_id: int = 0
    initial_exec_mask: int = FULL_EXEC_MASK
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None
    branch_outcomes: Optional[Mapping[int, bool]] = None
    forwarded_pcs: frozenset[int] = frozenset()
    initial_sgpr: Optional[Mapping[int, int]] = None
    initial_vgpr: Optional[Mapping[int, Sequence[int]]] = None
    workgroup_id: Tuple[int, int, int] = (0, 0, 0)
    wave_index_in_wg: int = 0


def _decoded_to_instructions(
    decoded: Sequence[DecodedInsn],
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
    branch_outcomes: Optional[Mapping[int, bool]] = None,
    forwarded_pcs: Optional[Set[int]] = None,
) -> List[Instruction]:
    bo = branch_outcomes or {}
    fp = forwarded_pcs or set()
    barrier_seq = 0
    out: List[Instruction] = []
    for d in decoded:
        b_id = 0
        if d.pipe is Pipe.BARRIER:
            b_id = barrier_seq
            barrier_seq += 1
        out.append(d.to_instruction(
            vgpr_state=vgpr_state,
            taken=bool(bo.get(d.pc, False)),
            forwarded=(d.pc in fp),
            barrier_id=b_id,
        ))
    return out


def _coerce_program(
    program: ProgramInput,
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
    branch_outcomes: Optional[Mapping[int, bool]] = None,
    forwarded_pcs: Optional[Set[int]] = None,
) -> List[Instruction]:
    if isinstance(program, AMDGPUKernel):
        return _decoded_to_instructions(
            decode_gfx11(program.text_bytes, base_pc=program.text_addr, stop_at_code_end=True),
            vgpr_state, branch_outcomes, forwarded_pcs,
        )
    if isinstance(program, (bytes, bytearray, memoryview, str)):
        return _decoded_to_instructions(decode_gfx11(program), vgpr_state, branch_outcomes, forwarded_pcs)
    prog_list = list(program)
    if not prog_list:
        return []
    if isinstance(prog_list[0], DecodedInsn):
        return _decoded_to_instructions(prog_list, vgpr_state, branch_outcomes, forwarded_pcs)  # type: ignore[arg-type]
    return prog_list  # type: ignore[return-value]


@dataclass(frozen=True)
class TraceSample:
    """Observed trace sample used by `fit_profile()` for latency calibration."""

    program: ProgramInput
    observed: ObservedInput
    contention: Tuple[ContentionEvent, ...] = ()
    initial_exec_mask: int = FULL_EXEC_MASK
    wave_id: int = 0
    simd_id: int = 0
    format: str = "auto"  # "auto" | "sqtt" | "sqtt_hw" | "tuple"
    sqtt_format: bool = True
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None
    branch_outcomes: Optional[Mapping[int, bool]] = None
    initial_sgpr: Optional[Mapping[int, int]] = None
    initial_vgpr: Optional[Mapping[int, Sequence[int]]] = None
    hw_sqtt_mode: Optional[bool] = None

    def resolved_format(self) -> str:
        if self.format != "auto":
            return self.format
        if not self.sqtt_format:
            return "tuple"
        if isinstance(self.observed, (bytes, bytearray, memoryview, str)):
            return "sqtt_hw"
        obs_list = list(self.observed)
        if obs_list and isinstance(obs_list[0], (SQTTPacket, SQTTAlignedEvent)):
            return "sqtt_hw"
        return "sqtt"


def lds_bank(addr: int) -> int:
    """Return 32-bank Shared SRAM bank index `(addr >> 2) & 31` for a byte address."""
    return (addr >> LDS_BANK_SHIFT) & (LDS_NUM_BANKS - 1)


def _half_wave_bank_counts(
    addrs: Sequence[int],
    exec_mask: int,
    lane_start: int,
    lane_end: int,
    byte_off: int = 0,
) -> Tuple[int, Tuple[int, ...]]:
    """Compute per-bank distinct dword counts for one 16-lane half-wave."""
    bank_dwords: List[Set[int]] = [set() for _ in range(LDS_NUM_BANKS)]
    for lane in range(lane_start, lane_end):
        if (exec_mask >> lane) & 1:
            dword = (addrs[lane] + byte_off) >> LDS_BANK_SHIFT
            bank = dword & (LDS_NUM_BANKS - 1)
            bank_dwords[bank].add(dword)
    counts = tuple(len(s) for s in bank_dwords)
    max_way = max(counts) if counts else 0
    return max_way, counts


def analyze_lds_access(
    pipe: Pipe,
    addrs: Optional[Sequence[int]],
    exec_mask: int = FULL_EXEC_MASK,
    profile: Optional[LatencyProfile] = None,
    lds_dwords: int = 1,
    lds_sub_offsets: Optional[Sequence[int]] = None,
) -> LDSBankAnalysis:
    """Analyze byte-addressed 32-bank LDS bank conflicts for LDS_RD or LDS_WR_1..5."""
    prof = profile or LatencyProfile()
    mask = exec_mask & FULL_EXEC_MASK
    if addrs is None:
        lane_addrs: Sequence[int] = tuple(i * 4 for i in range(WAVE_LANES))
    else:
        if len(addrs) != WAVE_LANES:
            raise ValueError(f"LDS address vector must have {WAVE_LANES} lanes, got {len(addrs)}")
        lane_addrs = addrs

    if mask == 0:
        zeros = (0,) * LDS_NUM_BANKS
        return LDSBankAnalysis(0, 0, 0, 0, 0, zeros, zeros)

    if pipe is Pipe.LDS_RD:
        dwords = max(1, lds_dwords)
        offsets = tuple(lds_sub_offsets) if lds_sub_offsets is not None else tuple(d * 4 for d in range(dwords))
        total_extra = 0
        max_h0, max_h1 = 0, 0
        c0_first = (0,) * LDS_NUM_BANKS
        c1_first = (0,) * LDS_NUM_BANKS
        for idx, byte_off in enumerate(offsets):
            h0, c0 = _half_wave_bank_counts(lane_addrs, mask, 0, HALF_WAVE_LANES, byte_off)
            h1, c1 = _half_wave_bank_counts(lane_addrs, mask, HALF_WAVE_LANES, WAVE_LANES, byte_off)
            if idx == 0:
                c0_first, c1_first = c0, c1
            max_h0 = max(max_h0, h0)
            max_h1 = max(max_h1, h1)
            total_extra += max(0, h0 - 1) + max(0, h1 - 1)
        port_cycles = prof.lds_rd_base * len(offsets) + total_extra
        return LDSBankAnalysis(
            half0_max_way=max_h0, half1_max_way=max_h1,
            conflict_extra=total_extra, conflict_factor=max(1, max_h0, max_h1),
            port_cycles=port_cycles,
            distinct_dwords_per_bank_half0=c0_first, distinct_dwords_per_bank_half1=c1_first,
        )

    if pipe in LDS_WR_PIPES:
        dwords = max(LDS_WR_PIPES[pipe], lds_dwords)
        offsets = tuple(lds_sub_offsets) if lds_sub_offsets is not None else tuple(d * 4 for d in range(dwords))
        total_port = 0
        max_h0, max_h1 = 0, 0
        c0_first = (0,) * LDS_NUM_BANKS
        c1_first = (0,) * LDS_NUM_BANKS
        for idx, byte_off in enumerate(offsets):
            h0, c0 = _half_wave_bank_counts(lane_addrs, mask, 0, HALF_WAVE_LANES, byte_off)
            h1, c1 = _half_wave_bank_counts(lane_addrs, mask, HALF_WAVE_LANES, WAVE_LANES, byte_off)
            if idx == 0:
                c0_first, c1_first = c0, c1
            max_h0 = max(max_h0, h0)
            max_h1 = max(max_h1, h1)
            total_port += max(1, h0, h1)
        return LDSBankAnalysis(
            half0_max_way=max_h0, half1_max_way=max_h1,
            conflict_extra=total_port - len(offsets), conflict_factor=max(1, max_h0, max_h1),
            port_cycles=total_port,
            distinct_dwords_per_bank_half0=c0_first, distinct_dwords_per_bank_half1=c1_first,
        )

    raise ValueError(f"analyze_lds_access called on non-LDS pipe: {pipe!r}")


# ---------------------------------------------------------------------------
# Real 32-Lane Wave32 Functional ISA Execution Engine
# ---------------------------------------------------------------------------

def _u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _f32_to_u32(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", float(f)))[0]
    except OverflowError:
        return 0x7F800000 if f > 0 else 0xFF800000


def _i32(u: int) -> int:
    u &= 0xFFFFFFFF
    return u - 0x100000000 if (u & 0x80000000) else u


def _i64(u: int) -> int:
    u &= 0xFFFFFFFFFFFFFFFF
    return u - 0x10000000000000000 if (u & 0x8000000000000000) else u


def _eval_cmp_cond(cond: str, a: int, b: int, dtype: str) -> bool:
    if dtype == "f32":
        fa, fb = _u32_to_f32(a), _u32_to_f32(b)
        nan = math.isnan(fa) or math.isnan(fb)
        if cond == "f": return False
        if cond == "t": return True
        if cond == "o": return not nan
        if cond == "u": return nan
        if cond == "lt": return (not nan) and (fa < fb)
        if cond == "eq": return (not nan) and (fa == fb)
        if cond == "le": return (not nan) and (fa <= fb)
        if cond == "gt": return (not nan) and (fa > fb)
        if cond == "lg": return (not nan) and (fa != fb)
        if cond == "ge": return (not nan) and (fa >= fb)
        if cond == "nge": return nan or not (fa >= fb)
        if cond == "nlg": return nan or (fa == fb)
        if cond == "ngt": return nan or not (fa > fb)
        if cond == "nle": return nan or not (fa <= fb)
        if cond == "neq": return nan or (fa != fb)
        if cond == "nlt": return nan or not (fa < fb)
        return False
    if dtype == "i32":
        sa, sb = _i32(a), _i32(b)
    elif dtype == "u32":
        sa, sb = a & 0xFFFFFFFF, b & 0xFFFFFFFF
    elif dtype == "i64":
        sa, sb = _i64(a), _i64(b)
    else:
        sa, sb = a & 0xFFFFFFFFFFFFFFFF, b & 0xFFFFFFFFFFFFFFFF
    return {
        "f": False, "t": True, "lt": sa < sb, "eq": sa == sb, "le": sa <= sb,
        "gt": sa > sb, "ne": sa != sb, "lg": sa != sb, "ge": sa >= sb,
    }.get(cond, False)


class Wave32ISAState:
    """Real 32-lane GFX11 Wave32 architectural state (SGPR, VGPR, EXEC, VCC, SCC, LDS, memory).

    Executes decoded GFX11 instructions so per-lane LDS addresses, EXEC masks,
    conditional branches, loops, and data transfers emerge directly from the binary ISA.
    """

    def __init__(
        self,
        exec_mask: int = FULL_EXEC_MASK,
        initial_sgpr: Optional[Mapping[int, int]] = None,
        initial_vgpr: Optional[Mapping[int, Sequence[int]]] = None,
        lds_mem: Optional[bytearray] = None,
        global_mem: Optional[Dict[int, int]] = None,
        wave_index_in_wg: int = 0,
        workgroup_id: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.sgpr: List[int] = [0] * 106
        self._zero_lanes: Tuple[int, ...] = (0,) * WAVE_LANES
        base_tid = wave_index_in_wg * WAVE_LANES
        self.vgpr: Dict[int, List[int]] = {
            0: [(base_tid + lane) & 0xFFFFFFFF for lane in range(WAVE_LANES)]
        }
        self.sgpr[14] = workgroup_id[0] & 0xFFFFFFFF
        self.sgpr[15] = workgroup_id[1] & 0xFFFFFFFF
        self.sgpr[16] = workgroup_id[2] & 0xFFFFFFFF
        if initial_sgpr:
            for idx, val in initial_sgpr.items():
                if 0 <= idx < 106:
                    self.sgpr[idx] = int(val) & 0xFFFFFFFF
        if initial_vgpr:
            for idx, lanes in initial_vgpr.items():
                if 0 <= idx < 256:
                    if len(lanes) != WAVE_LANES:
                        raise ValueError(f"initial_vgpr[{idx}] must have {WAVE_LANES} lanes")
                    self.vgpr[idx] = [int(v) & 0xFFFFFFFF for v in lanes]
        self.exec_lo: int = exec_mask & FULL_EXEC_MASK
        self.vcc_lo: int = 0
        self.scc: int = 0
        self._lds_mem: Optional[bytearray] = lds_mem
        self.global_mem: Dict[int, int] = global_mem if global_mem is not None else {}

    @property
    def lds(self) -> bytearray:
        if self._lds_mem is None:
            self._lds_mem = bytearray(65536)
        return self._lds_mem

    def _get_vgpr(self, idx: int) -> List[int]:
        v = self.vgpr.get(idx)
        return list(v) if v is not None else [0] * WAVE_LANES

    def read_ssrc(self, code: int, literal: Optional[int] = None) -> int:
        if 0 <= code <= 105:
            return self.sgpr[code]
        if code == VCC_LO_ENC:
            return self.vcc_lo
        if code == EXEC_LO_ENC:
            return self.exec_lo
        if code == NULL_ENC:
            return 0
        if code == LITERAL_ENC:
            return 0 if literal is None else (literal & 0xFFFFFFFF)
        inl = _decode_inline_u32(code)
        return 0 if inl is None else inl

    def read_ssrc64(self, code: int, literal: Optional[int] = None) -> int:
        if 0 <= code <= 104:
            return self.sgpr[code] | (self.sgpr[code + 1] << 32)
        return self.read_ssrc(code, literal)

    def write_sdst(self, code: int, val: int) -> None:
        v = val & 0xFFFFFFFF
        if 0 <= code <= 105:
            self.sgpr[code] = v
        elif code == VCC_LO_ENC:
            self.vcc_lo = v
        elif code == EXEC_LO_ENC:
            self.exec_lo = v

    def write_sdst64(self, code: int, val: int) -> None:
        if 0 <= code <= 104:
            self.sgpr[code] = val & 0xFFFFFFFF
            self.sgpr[code + 1] = (val >> 32) & 0xFFFFFFFF
        else:
            self.write_sdst(code, val)

    def read_vsrc_lanes(self, code: int, literal: Optional[int] = None) -> List[int]:
        if 256 <= code <= 511:
            return self._get_vgpr(code - 256)
        sv = self.read_ssrc(code, literal)
        return [sv] * WAVE_LANES

    def read_vsrc64_lanes(self, code: int, literal: Optional[int] = None) -> List[int]:
        if 256 <= code <= 510:
            lo = self._get_vgpr(code - 256)
            hi = self._get_vgpr(code - 255)
            return [lo[i] | (hi[i] << 32) for i in range(WAVE_LANES)]
        sv = self.read_ssrc64(code, literal)
        return [sv] * WAVE_LANES

    def write_vdst_lanes(self, vreg: int, vals: Sequence[int]) -> None:
        if not (0 <= vreg < 256):
            return
        dst = self.vgpr.get(vreg)
        if dst is None:
            dst = [0] * WAVE_LANES
            self.vgpr[vreg] = dst
        mask = self.exec_lo
        for lane in range(WAVE_LANES):
            if (mask >> lane) & 1:
                dst[lane] = vals[lane] & 0xFFFFFFFF

    def compute_lds_addrs(self, d: DecodedInsn) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Compute live 32-lane byte addresses from VGPR[addr] + offset for DS ops."""
        addr_v = d.src_codes[0] if d.src_codes else 0
        base_lanes = self.vgpr.get(addr_v, self._zero_lanes) if 0 <= addr_v < 256 else [lane * 4 for lane in range(WAVE_LANES)]
        if d.lds_is_2addr:
            stride = d.lds_2addr_stride
            if d.lds_dwords == 2:
                sub = (d.lds_offset0 * stride, d.lds_offset1 * stride)
            else:
                o0, o1 = d.lds_offset0 * stride, d.lds_offset1 * stride
                sub = (o0, o0 + 4, o1, o1 + 4)
            return tuple(a & 0xFFFF for a in base_lanes), sub
        return tuple((a + d.lds_offset) & 0xFFFF for a in base_lanes), None

    def eval_branch(self, d: DecodedInsn) -> bool:
        """Evaluate exact GFX11 SOPP branch condition from live SCC / VCC / EXEC."""
        op = d.op_code
        if op == 0x20:  # s_branch
            return True
        if op == 0x21:  # s_cbranch_scc0
            return self.scc == 0
        if op == 0x22:  # s_cbranch_scc1
            return self.scc != 0
        if op == 0x23:  # s_cbranch_vccz
            return (self.vcc_lo & FULL_EXEC_MASK) == 0
        if op == 0x24:  # s_cbranch_vccnz
            return (self.vcc_lo & FULL_EXEC_MASK) != 0
        if op == 0x25:  # s_cbranch_execz
            return (self.exec_lo & FULL_EXEC_MASK) == 0
        if op == 0x26:  # s_cbranch_execnz
            return (self.exec_lo & FULL_EXEC_MASK) != 0
        return False

    def execute(self, d: DecodedInsn) -> Optional[bool]:
        """Execute one decoded GFX11 instruction on architectural state.

        Returns branch-taken bool for `Pipe.JUMP`, else `None`.
        """
        enc = d.encoding
        op = d.op_code
        lit = d.literal

        if d.pipe is Pipe.JUMP:
            if enc == "SOPK" and op == 0x14:  # s_call_b64
                self.write_sdst64(d.dst_code, d.pc + d.size)
                return True
            return self.eval_branch(d)

        if enc == "SOP2":
            s0 = self.read_ssrc(d.src_codes[0], lit)
            s1 = self.read_ssrc(d.src_codes[1], lit)
            if op == 0x00:  # s_add_u32
                res = s0 + s1; self.write_sdst(d.dst_code, res); self.scc = 1 if res >= 0x100000000 else 0
            elif op == 0x01:  # s_sub_u32
                res = s0 - s1; self.write_sdst(d.dst_code, res); self.scc = 1 if res < 0 else 0
            elif op == 0x02:  # s_add_i32
                r = _i32(s0) + _i32(s1); self.write_sdst(d.dst_code, r); self.scc = 1 if (r < -0x80000000 or r > 0x7FFFFFFF) else 0
            elif op == 0x03:  # s_sub_i32
                r = _i32(s0) - _i32(s1); self.write_sdst(d.dst_code, r); self.scc = 1 if (r < -0x80000000 or r > 0x7FFFFFFF) else 0
            elif op == 0x04:  # s_addc_u32
                res = s0 + s1 + self.scc; self.write_sdst(d.dst_code, res); self.scc = 1 if res >= 0x100000000 else 0
            elif op == 0x05:  # s_subb_u32
                res = s0 - s1 - self.scc; self.write_sdst(d.dst_code, res); self.scc = 1 if res < 0 else 0
            elif op == 0x08:  # s_lshl_b32
                res = (s0 << (s1 & 31)) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x09:  # s_lshl_b64
                v64 = self.read_ssrc64(d.src_codes[0], lit)
                res64 = (v64 << (s1 & 63)) & 0xFFFFFFFFFFFFFFFF; self.write_sdst64(d.dst_code, res64); self.scc = int(res64 != 0)
            elif op == 0x0A:  # s_lshr_b32
                res = (s0 >> (s1 & 31)) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x0B:  # s_lshr_b64
                v64 = self.read_ssrc64(d.src_codes[0], lit)
                res64 = (v64 >> (s1 & 63)) & 0xFFFFFFFFFFFFFFFF; self.write_sdst64(d.dst_code, res64); self.scc = int(res64 != 0)
            elif op == 0x0C:  # s_ashr_i32
                res = (_i32(s0) >> (s1 & 31)) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x0D:  # s_ashr_i64
                v64 = _i64(self.read_ssrc64(d.src_codes[0], lit))
                res64 = (v64 >> (s1 & 63)) & 0xFFFFFFFFFFFFFFFF; self.write_sdst64(d.dst_code, res64); self.scc = int(res64 != 0)
            elif 0x0E <= op <= 0x11:  # s_lshl{1..4}_add_u32
                sh = op - 0x0D; res = ((s0 << sh) + s1) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res)
            elif op == 0x16:  # s_and_b32
                res = s0 & s1; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x18:  # s_or_b32
                res = s0 | s1; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x1A:  # s_xor_b32
                res = s0 ^ s1; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op == 0x2C:  # s_mul_i32
                res = (_i32(s0) * _i32(s1)) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res)
            elif op == 0x30:  # s_cselect_b32
                self.write_sdst(d.dst_code, s0 if self.scc else s1)
            return None

        if enc == "SOP1":
            s0 = self.read_ssrc(d.src_codes[0], lit)
            if op == 0x00:  # s_mov_b32
                self.write_sdst(d.dst_code, s0)
            elif op == 0x01:  # s_mov_b64
                self.write_sdst64(d.dst_code, self.read_ssrc64(d.src_codes[0], lit))
            elif op == 0x02:  # s_cmov_b32
                if self.scc:
                    self.write_sdst(d.dst_code, s0)
            elif op == 0x1E:  # s_not_b32
                res = (~s0) & 0xFFFFFFFF; self.write_sdst(d.dst_code, res); self.scc = int(res != 0)
            elif op in _SOP1_SAVEEXEC_OPS:
                old_exec = self.exec_lo
                if op in (0x20, 0x21): new_exec = s0 & old_exec
                elif op in (0x22, 0x23): new_exec = s0 | old_exec
                elif op in (0x24, 0x25): new_exec = s0 ^ old_exec
                elif op in (0x26, 0x27): new_exec = s0 & (~old_exec)
                elif op in (0x28, 0x29): new_exec = s0 | (~old_exec)
                elif op in (0x2A, 0x2B): new_exec = ~(s0 & old_exec)
                elif op in (0x2C, 0x2D): new_exec = ~(s0 | old_exec)
                elif op in (0x2E, 0x2F): new_exec = ~(s0 ^ old_exec)
                elif op in (0x30, 0x31): new_exec = (~s0) & old_exec
                else: new_exec = (~s0) | old_exec
                self.write_sdst(d.dst_code, old_exec)
                self.exec_lo = new_exec & FULL_EXEC_MASK
                self.scc = int(self.exec_lo != 0)
            return None

        if enc == "SOPK":
            simm = d.simm16 or 0
            uimm = simm & 0xFFFF
            cur = self.read_ssrc(d.dst_code)
            if op == 0x00:  # s_movk_i32
                self.write_sdst(d.dst_code, simm & 0xFFFFFFFF)
            elif op == 0x02:  # s_cmovk_i32
                if self.scc:
                    self.write_sdst(d.dst_code, simm & 0xFFFFFFFF)
            elif op == 0x0F:  # s_addk_i32
                r = _i32(cur) + simm; self.write_sdst(d.dst_code, r); self.scc = 1 if (r < -0x80000000 or r > 0x7FFFFFFF) else 0
            elif op == 0x10:  # s_mulk_i32
                self.write_sdst(d.dst_code, _i32(cur) * simm)
            elif op in _SOPK_CMP_OPS:
                cond_map = {
                    0x03: ("eq", "i32"), 0x04: ("lg", "i32"), 0x05: ("gt", "i32"),
                    0x06: ("ge", "i32"), 0x07: ("lt", "i32"), 0x08: ("le", "i32"),
                    0x09: ("eq", "u32"), 0x0A: ("lg", "u32"), 0x0B: ("gt", "u32"),
                    0x0C: ("ge", "u32"), 0x0D: ("lt", "u32"), 0x0E: ("le", "u32"),
                }
                cond, dt = cond_map[op]
                rhs = simm & 0xFFFFFFFF if dt == "i32" else uimm
                self.scc = int(_eval_cmp_cond(cond, cur, rhs, dt))
            return None

        if enc == "SOPC":
            s0 = self.read_ssrc(d.src_codes[0], lit)
            s1 = self.read_ssrc(d.src_codes[1], lit)
            cond_map = {
                0x00: ("eq", "i32"), 0x01: ("lg", "i32"), 0x02: ("gt", "i32"),
                0x03: ("ge", "i32"), 0x04: ("lt", "i32"), 0x05: ("le", "i32"),
                0x06: ("eq", "u32"), 0x07: ("lg", "u32"), 0x08: ("gt", "u32"),
                0x09: ("ge", "u32"), 0x0A: ("lt", "u32"), 0x0B: ("le", "u32"),
            }
            if op in cond_map:
                cond, dt = cond_map[op]
                self.scc = int(_eval_cmp_cond(cond, s0, s1, dt))
            return None

        if enc in ("VOPC", "VOPCX", "VOP3_VOPC", "VOP3_VOPCX"):
            is_x = enc in ("VOPCX", "VOP3_VOPCX")
            parts = d.mnemonic.split("_")
            # e.g. v_cmpx_eq_u32_e32 -> cond="eq", dtype="u32"
            cond = parts[2] if len(parts) >= 4 else "lt"
            dtype = parts[3] if len(parts) >= 4 else "i32"
            is64 = dtype in ("i64", "u64", "f64")
            a_lanes = self.read_vsrc64_lanes(d.src_codes[0], lit) if is64 else self.read_vsrc_lanes(d.src_codes[0], lit)
            b_lanes = self.read_vsrc64_lanes(d.src_codes[1], lit) if is64 else self.read_vsrc_lanes(d.src_codes[1], lit)
            out_mask = 0
            for lane in range(WAVE_LANES):
                if (self.exec_lo >> lane) & 1:
                    if _eval_cmp_cond(cond, a_lanes[lane], b_lanes[lane], dtype):
                        out_mask |= (1 << lane)
            if is_x:
                self.exec_lo = out_mask & FULL_EXEC_MASK
            else:
                self.write_sdst(d.dst_code, out_mask)
            return None

        if enc == "VOP1":
            a = self.read_vsrc_lanes(d.src_codes[0], lit)
            if op == 0x01:  # v_mov_b32
                self.write_vdst_lanes(d.dst_code, a)
            elif op == 0x02:  # v_readfirstlane_b32
                first = next((i for i in range(WAVE_LANES) if (self.exec_lo >> i) & 1), 0)
                self.write_sdst(d.dst_code, a[first])
            elif op == 0x37:  # v_not_b32
                self.write_vdst_lanes(d.dst_code, [(~x) & 0xFFFFFFFF for x in a])
            elif op == 0x05:  # v_cvt_f32_i32
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_i32(x)) for x in a])
            elif op == 0x06:  # v_cvt_f32_u32
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(x) for x in a])
            return None

        if enc == "VOP2":
            a = self.read_vsrc_lanes(d.src_codes[0], lit)
            b = self.read_vsrc_lanes(d.src_codes[1], lit)
            if op == 0x01:  # v_cndmask_b32
                self.write_vdst_lanes(d.dst_code, [b[i] if ((self.vcc_lo >> i) & 1) else a[i] for i in range(WAVE_LANES)])
            elif op == 0x03:  # v_add_f32
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_u32_to_f32(a[i]) + _u32_to_f32(b[i])) for i in range(WAVE_LANES)])
            elif op == 0x04:  # v_sub_f32
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_u32_to_f32(a[i]) - _u32_to_f32(b[i])) for i in range(WAVE_LANES)])
            elif op == 0x08:  # v_mul_f32
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_u32_to_f32(a[i]) * _u32_to_f32(b[i])) for i in range(WAVE_LANES)])
            elif op == 0x0B:  # v_mul_u32_u24
                self.write_vdst_lanes(d.dst_code, [((a[i] & 0xFFFFFF) * (b[i] & 0xFFFFFF)) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x18:  # v_lshlrev_b32 (dst = src1 << (src0 & 31))
                self.write_vdst_lanes(d.dst_code, [(b[i] << (a[i] & 31)) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x19:  # v_lshrrev_b32
                self.write_vdst_lanes(d.dst_code, [(b[i] >> (a[i] & 31)) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x1A:  # v_ashrrev_i32
                self.write_vdst_lanes(d.dst_code, [(_i32(b[i]) >> (a[i] & 31)) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x1B:  # v_and_b32
                self.write_vdst_lanes(d.dst_code, [a[i] & b[i] for i in range(WAVE_LANES)])
            elif op == 0x1C:  # v_or_b32
                self.write_vdst_lanes(d.dst_code, [a[i] | b[i] for i in range(WAVE_LANES)])
            elif op == 0x1D:  # v_xor_b32
                self.write_vdst_lanes(d.dst_code, [a[i] ^ b[i] for i in range(WAVE_LANES)])
            elif op == 0x20:  # v_add_co_ci_u32
                out_v = [0] * WAVE_LANES
                new_vcc = 0
                for i in range(WAVE_LANES):
                    if (self.exec_lo >> i) & 1:
                        s = a[i] + b[i] + ((self.vcc_lo >> i) & 1)
                        out_v[i] = s & 0xFFFFFFFF
                        if s >= 0x100000000:
                            new_vcc |= (1 << i)
                self.write_vdst_lanes(d.dst_code, out_v)
                self.vcc_lo = new_vcc
            elif op == 0x25:  # v_add_nc_u32
                self.write_vdst_lanes(d.dst_code, [(a[i] + b[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x26:  # v_sub_nc_u32
                self.write_vdst_lanes(d.dst_code, [(a[i] - b[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x2B:  # v_fmac_f32
                c = self._get_vgpr(d.dst_code)
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_u32_to_f32(a[i]) * _u32_to_f32(b[i]) + _u32_to_f32(c[i])) for i in range(WAVE_LANES)])
            return None

        if enc == "VOPD":
            opx, opy = (op >> 8) & 0xFF, op & 0xFF
            vdstx, vdsty = (d.dst_code >> 8) & 0xFF, d.dst_code & 0xFF
            sx0 = self.read_vsrc_lanes(d.src_codes[0], lit)
            sx1 = self.read_vsrc_lanes(d.src_codes[1], lit)
            sy0 = self.read_vsrc_lanes(d.src_codes[2], lit)
            sy1 = self.read_vsrc_lanes(d.src_codes[3], lit)
            old_dx = self._get_vgpr(vdstx)
            old_dy = self._get_vgpr(vdsty)

            def _eval_dual(dop: int, p0: List[int], p1: List[int], acc: List[int]) -> List[int]:
                if dop == 0x08: return list(p0)
                if dop == 0x04: return [_f32_to_u32(_u32_to_f32(p0[i]) + _u32_to_f32(p1[i])) for i in range(WAVE_LANES)]
                if dop == 0x03: return [_f32_to_u32(_u32_to_f32(p0[i]) * _u32_to_f32(p1[i])) for i in range(WAVE_LANES)]
                if dop == 0x00: return [_f32_to_u32(_u32_to_f32(p0[i]) * _u32_to_f32(p1[i]) + _u32_to_f32(acc[i])) for i in range(WAVE_LANES)]
                if dop == 0x10: return [(p0[i] + p1[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)]
                if dop == 0x11: return [(p1[i] << (p0[i] & 31)) & 0xFFFFFFFF for i in range(WAVE_LANES)]
                return acc

            self.write_vdst_lanes(vdstx, _eval_dual(opx, sx0, sx1, old_dx))
            self.write_vdst_lanes(vdsty, _eval_dual(opy, sy0, sy1, old_dy))
            return None

        if enc in ("VOP3", "VOP3SD"):
            if op == 0x101:  # v_cndmask_b32_e64
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                cmask = self.read_ssrc(d.src_codes[2], lit)
                self.write_vdst_lanes(d.dst_code, [b[i] if ((cmask >> i) & 1) else a[i] for i in range(WAVE_LANES)])
            elif op == 0x246:  # v_lshl_add_u32
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                c = self.read_vsrc_lanes(d.src_codes[2], lit)
                self.write_vdst_lanes(d.dst_code, [((a[i] << (b[i] & 31)) + c[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x256:  # v_lshl_or_b32
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                c = self.read_vsrc_lanes(d.src_codes[2], lit)
                self.write_vdst_lanes(d.dst_code, [((a[i] << (b[i] & 31)) | c[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x213:  # v_fma_f32
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                c = self.read_vsrc_lanes(d.src_codes[2], lit)
                self.write_vdst_lanes(d.dst_code, [_f32_to_u32(_u32_to_f32(a[i]) * _u32_to_f32(b[i]) + _u32_to_f32(c[i])) for i in range(WAVE_LANES)])
            elif op == 0x300:  # v_add_co_u32
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                out_v = [0] * WAVE_LANES
                carry = 0
                for i in range(WAVE_LANES):
                    if (self.exec_lo >> i) & 1:
                        s = a[i] + b[i]
                        out_v[i] = s & 0xFFFFFFFF
                        if s >= 0x100000000:
                            carry |= (1 << i)
                self.write_vdst_lanes(d.dst_code, out_v)
                self.write_sdst(d.sdst_code, carry)
            elif op == 0x32C:  # v_mul_lo_u32
                a = self.read_vsrc_lanes(d.src_codes[0], lit)
                b = self.read_vsrc_lanes(d.src_codes[1], lit)
                self.write_vdst_lanes(d.dst_code, [(a[i] * b[i]) & 0xFFFFFFFF for i in range(WAVE_LANES)])
            elif op == 0x33C:  # v_lshlrev_b64
                sh = self.read_vsrc_lanes(d.src_codes[0], lit)
                v64 = self.read_vsrc64_lanes(d.src_codes[1], lit)
                res = [(v64[i] << (sh[i] & 63)) & 0xFFFFFFFFFFFFFFFF for i in range(WAVE_LANES)]
                self.write_vdst_lanes(d.dst_code, [x & 0xFFFFFFFF for x in res])
                self.write_vdst_lanes(d.dst_code + 1, [(x >> 32) & 0xFFFFFFFF for x in res])
            return None

        if enc == "DS":
            addrs, sub_offs = self.compute_lds_addrs(d)
            data0_v = d.src_codes[1] if len(d.src_codes) > 1 else 0
            if d.pipe is Pipe.LDS_RD:
                offs = sub_offs if sub_offs is not None else tuple(4 * k for k in range(d.lds_dwords))
                for k, ob in enumerate(offs):
                    vals = [0] * WAVE_LANES
                    for lane in range(WAVE_LANES):
                        if (self.exec_lo >> lane) & 1:
                            a = (addrs[lane] + ob) & 0xFFFC
                            vals[lane] = struct.unpack_from("<I", self.lds, a)[0]
                    self.write_vdst_lanes(d.dst_code + k, vals)
            else:
                offs = sub_offs if sub_offs is not None else tuple(4 * k for k in range(d.lds_dwords))
                for k, ob in enumerate(offs):
                    src_v = self._get_vgpr((data0_v + k) & 0xFF)
                    for lane in range(WAVE_LANES):
                        if (self.exec_lo >> lane) & 1:
                            a = (addrs[lane] + ob) & 0xFFFC
                            struct.pack_into("<I", self.lds, a, src_v[lane] & 0xFFFFFFFF)
            return None

        if enc == "SMEM" and self.global_mem:
            base_ptr = self.read_ssrc64(d.src_codes[0])
            off_b = d.lds_offset
            if d.src_codes[1] != NULL_ENC:
                off_b += self.read_ssrc(d.src_codes[1])
            addr = base_ptr + off_b
            for k in range(len(d.dst)):
                if (addr + 4 * k) in self.global_mem:
                    self.write_sdst(d.dst_code + k, self.global_mem[addr + 4 * k])
            return None

        return None


# ---------------------------------------------------------------------------
# Shared CU Port Scheduler & Multi-Wave 4-SIMD32 Cycle Simulator
# ---------------------------------------------------------------------------

class SharedPortScheduler:
    """Exact cycle-occupancy calendar for a shared CU port (LDS or VMEM)."""

    def __init__(self, background: Sequence[ContentionEvent] = ()) -> None:
        self._busy: Set[int] = set()
        for ev in background:
            for c in range(ev.start_cycle, ev.start_cycle + ev.duration_cycles):
                self._busy.add(c)

    def peek_start(self, earliest: int, duration: int) -> int:
        if duration <= 0:
            return earliest
        c = earliest
        while c in self._busy:
            c += 1
        return c

    def reserve(self, earliest: int, duration: int) -> Tuple[int, int, int]:
        if duration <= 0:
            return earliest, earliest, 0
        start = self.peek_start(earliest, duration)
        c = start
        remaining = duration
        while remaining > 0:
            if c not in self._busy:
                self._busy.add(c)
                remaining -= 1
            c += 1
        end_excl = c
        stolen_mid = (end_excl - start) - duration
        return start, end_excl, stolen_mid


def _default_alu_src(pipe: Pipe, is_vopd: bool = False) -> AluSrc:
    if is_vopd:
        return AluSrc.VOPD
    if pipe in VALU_PIPES:
        return AluSrc.VALU
    if pipe in SALU_PIPES:
        return AluSrc.SALU
    if pipe in LDS_PIPES:
        return AluSrc.LDS
    if pipe is Pipe.SMEM_RD:
        return AluSrc.SMEM
    if pipe in VMEM_PIPES:
        return AluSrc.VMEM
    if pipe is Pipe.JUMP:
        return AluSrc.BRANCH
    if pipe is Pipe.BARRIER:
        return AluSrc.BARRIER
    if pipe is Pipe.WAIT_LGKMCNT:
        return AluSrc.WAITCNT
    return AluSrc.ENDPGM


def can_dual_issue_pair(first: Instruction, second: Instruction, profile: LatencyProfile) -> bool:
    """Return True iff `(first, second)` are eligible for VALU/SALU dual-issue."""
    if not profile.enable_dual_issue:
        return False
    if first.is_vopd or second.is_vopd:
        return False
    first_is_valu = first.pipe in VALU_PIPES and first.pipe is not Pipe.VALU1_WR_EXEC
    first_is_salu = first.pipe in SALU_PIPES
    second_is_valu = second.pipe in VALU_PIPES and second.pipe is not Pipe.VALU1_WR_EXEC
    second_is_salu = second.pipe in SALU_PIPES
    if not ((first_is_valu and second_is_salu) or (first_is_salu and second_is_valu)):
        return False
    dst1 = set(first.effective_dst())
    src1 = set(first.effective_src())
    dst2 = set(second.effective_dst())
    src2 = set(second.effective_src())
    if dst1 & src2:
        return False
    if dst1 & dst2:
        return False
    if src1 & dst2:
        return False
    return True


@dataclass
class _WaveState:
    wave_id: int
    simd_id: int
    program: List[Instruction]
    exec_mask: int = FULL_EXEC_MASK
    idx: int = 0
    next_fetch_cycle: int = 0
    valu_pipe_free: int = 0
    salu_pipe_free: int = 0
    reg_ready: Dict[str, Tuple[int, Pipe]] = field(default_factory=dict)
    reg_wb_done: Dict[str, int] = field(default_factory=dict)
    lgkm_completions: List[int] = field(default_factory=list)
    vmem_completions: List[int] = field(default_factory=list)
    vs_completions: List[int] = field(default_factory=list)
    exp_completions: List[int] = field(default_factory=list)
    waiting_barrier_id: Optional[int] = None
    barrier_arrival_cycle: Optional[int] = None
    events: List[CycleEvent] = field(default_factory=list)
    # Real ISA execution state + control-flow PC table + s_delay_alu pending hints
    isa: Optional[Wave32ISAState] = None
    pc_to_idx: Dict[int, int] = field(default_factory=dict)
    explicit_vgpr_override: Optional[Mapping[str, Sequence[int]]] = None
    explicit_branch_override: Optional[Mapping[int, bool]] = None
    recent_valu_issues: List[Tuple[int, int]] = field(default_factory=list)  # (issue_cyc, ready_cyc)
    recent_salu_issues: List[int] = field(default_factory=list)
    pending_delay_hints: Dict[int, int] = field(default_factory=dict)  # step_index -> min_issue_cycle
    step_counter: int = 0
    halted: bool = False

    def done(self) -> bool:
        return self.halted or self.idx >= len(self.program)

    def current_insn(self) -> Instruction:
        return self.program[self.idx]

    @staticmethod
    def _prune(queue: List[int], cycle: int) -> int:
        queue[:] = [c for c in queue if c > cycle]
        return len(queue)

    def prune_lgkm(self, cycle: int) -> int:
        return self._prune(self.lgkm_completions, cycle)

    def prune_vmem(self, cycle: int) -> int:
        return self._prune(self.vmem_completions, cycle)

    def prune_vs(self, cycle: int) -> int:
        return self._prune(self.vs_completions, cycle)

    @staticmethod
    def _wait_queue(queue: List[int], target_n: int, earliest: int, max_none: int) -> int:
        if target_n < 0:
            raise ValueError(f"waitcnt target must be >= 0, got {target_n}")
        if target_n >= max_none:
            return earliest
        pending = sorted(c for c in queue if c > earliest)
        if len(pending) <= target_n:
            return earliest
        must_complete = len(pending) - target_n
        return max(earliest, pending[must_complete - 1])

    def wait_ready_cycle(
        self, lgkm_target: int, vm_target: int, exp_target: int, vs_target: int, earliest: int
    ) -> int:
        t = self._wait_queue(self.lgkm_completions, lgkm_target, earliest, WAITCNT_LGKM_NONE)
        t = self._wait_queue(self.vmem_completions, vm_target, t, WAITCNT_VM_NONE)
        t = self._wait_queue(self.vs_completions, vs_target, t, WAITCNT_VS_NONE)
        t = self._wait_queue(self.exp_completions, exp_target, t, WAITCNT_EXP_NONE)
        return t


class Wave32Simulator:
    """Cycle-exact 4-SIMD32 Compute Unit, Real Wave32 ISA & 32-Bank LDS Simulator."""

    def __init__(
        self,
        profile: Optional[LatencyProfile] = None,
        contention: Sequence[ContentionEvent] = (),
        num_simds: int = CU_NUM_SIMDS,
        hw_sqtt_mode: bool = False,
        max_steps_per_wave: int = 10000,
    ) -> None:
        self.profile = profile or LatencyProfile()
        self.num_simds = num_simds
        self.contention = tuple(contention)
        self.hw_sqtt_mode = hw_sqtt_mode
        self.max_steps_per_wave = max_steps_per_wave
        self._lds_port = SharedPortScheduler(
            [e for e in self.contention if e.port is ContentionPort.OTHER_LDS]
        )
        self._vmem_port = SharedPortScheduler(
            [e for e in self.contention if e.port is ContentionPort.OTHER_VMEM]
        )
        self._simd_issue_free: List[int] = [0] * num_simds

    def _resolve_lds_addrs(
        self, wave: _WaveState, insn: Instruction
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """Use dynamic ISA VGPR addresses when available (unless explicitly overridden)."""
        if insn.pipe not in LDS_PIPES:
            return insn.lds_addrs, insn.lds_sub_offsets
        if (
            wave.isa is not None
            and insn.decoded is not None
            and insn.decoded.encoding == "DS"
            and (
                wave.explicit_vgpr_override is None
                or insn.decoded.lds_addr_reg not in wave.explicit_vgpr_override
            )
        ):
            return wave.isa.compute_lds_addrs(insn.decoded)
        return insn.lds_addrs, insn.lds_sub_offsets

    def _resolve_delay_hint(self, wave: _WaveState, instid: int) -> int:
        """Translate GFX11 `s_delay_alu` instid encoding into earliest ready cycle."""
        if instid == 0:
            return 0
        if 1 <= instid <= 4:  # VALU_DEP_1..4
            if len(wave.recent_valu_issues) >= instid:
                return wave.recent_valu_issues[-instid][1]
            return 0
        if 9 <= instid <= 11:  # SALU_CYCLE_1..3
            cyc_wait = instid - 8
            if wave.recent_salu_issues:
                return wave.recent_salu_issues[-1] + cyc_wait + 1
            return 0
        return 0

    def _apply_delay_alu(self, wave: _WaveState, insn: Instruction) -> None:
        d = insn.decoded
        if d is None:
            return
        if d.delay_instid0:
            t0 = self._resolve_delay_hint(wave, d.delay_instid0)
            if t0 > 0:
                key = wave.step_counter
                wave.pending_delay_hints[key] = max(wave.pending_delay_hints.get(key, 0), t0)
        if d.delay_instid1:
            t1 = self._resolve_delay_hint(wave, d.delay_instid1)
            if t1 > 0:
                skip = d.delay_instskip + 1  # NEXT=0 -> +1 insn after inst0
                key = wave.step_counter + skip
                wave.pending_delay_hints[key] = max(wave.pending_delay_hints.get(key, 0), t1)

    def _compute_ready_cycle(
        self,
        wave: _WaveState,
        insn: Instruction,
        earliest_fetch: int,
        ignore_simd_slot: bool = False,
    ) -> int:
        t = earliest_fetch
        if not ignore_simd_slot:
            t = max(t, self._simd_issue_free[wave.simd_id])

        # Honor any pending GFX11 s_delay_alu hint targeting this dynamic instruction
        if wave.step_counter in wave.pending_delay_hints:
            t = max(t, wave.pending_delay_hints[wave.step_counter])

        if insn.pipe in VALU_PIPES:
            t = max(t, wave.valu_pipe_free)
        elif insn.pipe in SALU_PIPES:
            t = max(t, wave.salu_pipe_free)

        # VALU32 1-cycle bypass applies ONLY to eligible VALU consumers,
        # NEVER across into SALU / DS / VMEM / SMEM / JUMP consumers!
        bypass_eligible = insn.pipe in VALU_PIPES
        is_mem_consumer = (insn.pipe in LDS_PIPES) or (insn.pipe in VMEM_PIPES)
        for reg in insn.effective_src():
            if reg in wave.reg_ready:
                prod_issue_or_ready, prod_pipe = wave.reg_ready[reg]
                if prod_pipe is Pipe.VALU32:
                    if is_mem_consumer and self.profile.valu_to_mem_latency > 0 and reg != EXEC_REG:
                        needed = prod_issue_or_ready + self.profile.valu_to_mem_latency
                    else:
                        needed = prod_issue_or_ready + self.profile.valu32_raw_latency(
                            insn.forwarded, consumer_eligible=bypass_eligible
                        )
                    t = max(t, needed)
                elif prod_pipe in SALU_PIPES and is_mem_consumer and self.profile.salu_to_mem_latency > 0 and reg != EXEC_REG:
                    t = max(t, prod_issue_or_ready + self.profile.salu_to_mem_latency)
                else:
                    t = max(t, prod_issue_or_ready)

        _, own_lat = self._estimate_issue_and_latency(wave, insn)
        for reg in insn.effective_dst():
            if reg in wave.reg_wb_done:
                prev_done = wave.reg_wb_done[reg]
                if t + own_lat < prev_done:
                    t = prev_done - own_lat

        if insn.pipe is Pipe.WAIT_LGKMCNT:
            t = wave.wait_ready_cycle(
                insn.lgkm_target, insn.vm_target, insn.exp_target, insn.vs_target, t
            )

        if insn.pipe in LDS_PIPES:
            mask = wave.exec_mask if insn.exec_mask is None else (insn.exec_mask & FULL_EXEC_MASK)
            lds_addrs, lds_subs = self._resolve_lds_addrs(wave, insn)
            lds_info = analyze_lds_access(
                insn.pipe, lds_addrs, mask, self.profile, insn.lds_dwords, lds_subs
            )
            if lds_info.port_cycles > 0:
                t = self._lds_port.peek_start(t, lds_info.port_cycles)
        elif insn.pipe in VMEM_PIPES:
            v_issue, _ = self.profile.pipe_timing(insn.pipe)
            t = self._vmem_port.peek_start(t, v_issue)

        return t

    def _estimate_issue_and_latency(self, wave: _WaveState, insn: Instruction) -> Tuple[int, int]:
        if insn.pipe in LDS_PIPES:
            mask = wave.exec_mask if insn.exec_mask is None else (insn.exec_mask & FULL_EXEC_MASK)
            lds_addrs, lds_subs = self._resolve_lds_addrs(wave, insn)
            info = analyze_lds_access(
                insn.pipe, lds_addrs, mask, self.profile, insn.lds_dwords, lds_subs
            )
            if insn.pipe is Pipe.LDS_RD:
                return info.port_cycles, self.profile.lds_rd_latency + info.conflict_extra
            return info.port_cycles, info.port_cycles + self.profile.lds_wr_latency_overhead
        return self.profile.pipe_timing(insn.pipe, taken=insn.taken)

    def _commit_instruction(
        self,
        wave: _WaveState,
        insn: Instruction,
        issue_cycle: int,
        earliest_fetch: int,
        alu_src: AluSrc,
        dual_issued: bool,
        reserve_simd: bool = True,
    ) -> Optional[CycleEvent]:
        # Handle non-SQTT scheduler hint `s_delay_alu`
        if insn.pipe is Pipe.DELAY_ALU:
            self._apply_delay_alu(wave, insn)
            wave.idx += 1
            return None

        # Pop consumed delay hint for this dynamic instruction
        wave.pending_delay_hints.pop(wave.step_counter, None)

        stall_cycles = max(0, issue_cycle - earliest_fetch)
        bank_conflicts = 0
        lds_port_cycles = 0
        port_contention_cycles = 0
        taken_flag = insn.taken

        # Execute real Wave32 ISA semantics (before advancing PC!)
        isa_branch_taken: Optional[bool] = None
        if wave.isa is not None and insn.decoded is not None:
            wave.isa.exec_lo = wave.exec_mask
            isa_branch_taken = wave.isa.execute(insn.decoded)
            if insn.pipe is Pipe.JUMP:
                if wave.explicit_branch_override is not None and insn.pc in wave.explicit_branch_override:
                    taken_flag = bool(wave.explicit_branch_override[insn.pc])
                elif isa_branch_taken is not None:
                    taken_flag = isa_branch_taken

        if insn.pipe in LDS_PIPES:
            mask = wave.exec_mask if insn.exec_mask is None else (insn.exec_mask & FULL_EXEC_MASK)
            lds_addrs, lds_subs = self._resolve_lds_addrs(wave, insn)
            info = analyze_lds_access(
                insn.pipe, lds_addrs, mask, self.profile, insn.lds_dwords, lds_subs
            )
            bank_conflicts = info.conflict_extra
            lds_port_cycles = info.port_cycles
            if info.port_cycles > 0:
                actual_start, end_excl, stolen_mid = self._lds_port.reserve(issue_cycle, info.port_cycles)
                if actual_start > issue_cycle:
                    stall_cycles += actual_start - issue_cycle
                    issue_cycle = actual_start
                port_contention_cycles = stolen_mid
                issue_occ = end_excl - issue_cycle
            else:
                issue_occ = 0
            if insn.pipe is Pipe.LDS_RD:
                duration = self.profile.lds_rd_latency + info.conflict_extra + port_contention_cycles
            else:
                duration = info.port_cycles + self.profile.lds_wr_latency_overhead + port_contention_cycles
            completion = issue_cycle + duration
            wave.lgkm_completions.append(completion)
            next_fetch_delta = max(1, issue_occ)
        elif insn.pipe in VMEM_PIPES:
            v_issue, v_lat = self.profile.pipe_timing(insn.pipe)
            actual_start, end_excl, stolen_mid = self._vmem_port.reserve(issue_cycle, v_issue)
            if actual_start > issue_cycle:
                stall_cycles += actual_start - issue_cycle
                issue_cycle = actual_start
            port_contention_cycles = stolen_mid
            issue_occ = end_excl - issue_cycle
            duration = v_lat + port_contention_cycles
            completion = issue_cycle + duration
            if insn.pipe is Pipe.VMEM_RD:
                wave.vmem_completions.append(completion)
            else:
                wave.vs_completions.append(completion)
            next_fetch_delta = max(1, issue_occ)
        elif insn.pipe is Pipe.SMEM_RD:
            # SMEM_RD increments lgkmcnt, NEVER traverses 32-bank LDS crossbar!
            s_issue, s_lat = self.profile.pipe_timing(Pipe.SMEM_RD)
            issue_occ = s_issue
            duration = s_lat
            completion = issue_cycle + duration
            wave.lgkm_completions.append(completion)
            next_fetch_delta = max(1, issue_occ)
        else:
            issue_occ, duration = self.profile.pipe_timing(insn.pipe, taken=taken_flag)
            completion = issue_cycle + duration
            if insn.pipe in VALU_PIPES:
                wave.valu_pipe_free = max(wave.valu_pipe_free, issue_cycle + issue_occ)
                wave.recent_valu_issues.append((issue_cycle, completion))
                if len(wave.recent_valu_issues) > 8:
                    wave.recent_valu_issues.pop(0)
                next_fetch_delta = 1
            elif insn.pipe in SALU_PIPES:
                wave.salu_pipe_free = max(wave.salu_pipe_free, issue_cycle + issue_occ)
                wave.recent_salu_issues.append(issue_cycle)
                if len(wave.recent_salu_issues) > 8:
                    wave.recent_salu_issues.pop(0)
                next_fetch_delta = 1
            elif insn.pipe is Pipe.JUMP:
                next_fetch_delta = issue_occ
            elif insn.pipe is Pipe.WAIT_LGKMCNT:
                next_fetch_delta = max(1, issue_occ) + self.profile.waitcnt_resume_gap
            else:
                next_fetch_delta = max(1, issue_occ)

        # Register scoreboard updates
        if insn.pipe in (Pipe.LDS_RD, Pipe.SMEM_RD, Pipe.VMEM_RD):
            for reg in insn.effective_dst():
                wave.reg_wb_done[reg] = max(wave.reg_wb_done.get(reg, 0), completion)
        else:
            for reg in insn.effective_dst():
                if insn.pipe is Pipe.VALU32:
                    wave.reg_ready[reg] = (issue_cycle, Pipe.VALU32)
                else:
                    wave.reg_ready[reg] = (completion, insn.pipe)
                wave.reg_wb_done[reg] = max(wave.reg_wb_done.get(reg, 0), completion)

        # Dynamic EXEC mask updates (from ISA execution or explicit Instruction.exec_mask)
        if insn.pipe in (Pipe.SALU_WR_EXEC, Pipe.VALU1_WR_EXEC):
            if insn.exec_mask is not None:
                wave.exec_mask = insn.exec_mask & FULL_EXEC_MASK
                if wave.isa is not None:
                    wave.isa.exec_lo = wave.exec_mask
            elif wave.isa is not None and insn.decoded is not None:
                wave.exec_mask = wave.isa.exec_lo & FULL_EXEC_MASK

        if reserve_simd:
            self._simd_issue_free[wave.simd_id] = max(self._simd_issue_free[wave.simd_id], issue_cycle + 1)

        wave.next_fetch_cycle = max(wave.next_fetch_cycle, issue_cycle + next_fetch_delta)
        lgkm_inflight = wave.prune_lgkm(issue_cycle)
        vm_inflight = wave.prune_vmem(issue_cycle)
        wave.prune_vs(issue_cycle)

        ev = CycleEvent(
            pc=insn.pc,
            issue_cycle=issue_cycle,
            stall_cycles=stall_cycles,
            duration_cycles=duration,
            wave_id=wave.wave_id,
            simd_id=wave.simd_id,
            alu_src=alu_src,
            pipe=insn.pipe,
            dual_issued=dual_issued or insn.is_vopd,
            bank_conflicts=bank_conflicts,
            lds_port_cycles=lds_port_cycles,
            port_contention_cycles=port_contention_cycles,
            lgkm_inflight=lgkm_inflight,
            vm_inflight=vm_inflight,
            completion_cycle=completion,
            exec_mask=wave.exec_mask,
            taken=taken_flag,
            mnemonic=insn.mnemonic,
        )

        # Real GFX11 hardware SQTT omits `s_endpgm` (emitting WAVEEND instead)
        if not (self.hw_sqtt_mode and insn.pipe is Pipe.ENDPGM):
            wave.events.append(ev)

        wave.step_counter += 1
        if wave.step_counter >= self.max_steps_per_wave or insn.pipe is Pipe.ENDPGM:
            wave.halted = True
            wave.idx = len(wave.program)
            return ev

        # Real PC-based branch control flow when branch target is in the decoded program
        if (
            insn.pipe is Pipe.JUMP
            and taken_flag
            and insn.decoded is not None
            and insn.decoded.branch_target is not None
            and insn.decoded.branch_target in wave.pc_to_idx
        ):
            wave.idx = wave.pc_to_idx[insn.decoded.branch_target]
        else:
            wave.idx += 1
        return ev

    def run_wave(
        self,
        program: ProgramInput,
        wave_id: int = 0,
        simd_id: int = 0,
        initial_exec_mask: int = FULL_EXEC_MASK,
        vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
        branch_outcomes: Optional[Mapping[int, bool]] = None,
        forwarded_pcs: Optional[Set[int]] = None,
        initial_sgpr: Optional[Mapping[int, int]] = None,
        initial_vgpr: Optional[Mapping[int, Sequence[int]]] = None,
        workgroup_id: Tuple[int, int, int] = (0, 0, 0),
        wave_index_in_wg: int = 0,
    ) -> List[CycleEvent]:
        spec = WaveSpec(
            program=program,
            wave_id=wave_id,
            simd_id=simd_id,
            initial_exec_mask=initial_exec_mask,
            vgpr_state=vgpr_state,
            branch_outcomes=branch_outcomes,
            forwarded_pcs=frozenset(forwarded_pcs or ()),
            initial_sgpr=initial_sgpr,
            initial_vgpr=initial_vgpr,
            workgroup_id=workgroup_id,
            wave_index_in_wg=wave_index_in_wg,
        )
        res = self.run_workgroup([spec])
        return res[wave_id]

    def run_workgroup(
        self,
        waves_spec: Sequence[Union[WaveSpec, Tuple[int, int, ProgramInput]]],
        initial_exec_masks: Optional[Mapping[int, int]] = None,
        vgpr_states: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
        branch_outcomes: Optional[Mapping[int, Mapping[int, bool]]] = None,
    ) -> Dict[int, List[CycleEvent]]:
        """Simulate multiple concurrent waves on a 4-SIMD32 CU with dynamic arbitration."""
        masks = dict(initial_exec_masks or {})
        vstates = dict(vgpr_states or {})
        boutcomes = dict(branch_outcomes or {})
        shared_lds: Optional[bytearray] = None
        waves: List[_WaveState] = []
        seen_ids: Set[int] = set()

        for idx, item in enumerate(waves_spec):
            if isinstance(item, WaveSpec):
                wid, sid, prog = item.wave_id, item.simd_id, item.program
                em = masks.get(wid, item.initial_exec_mask) & FULL_EXEC_MASK
                v_ov = item.vgpr_state if item.vgpr_state is not None else vstates.get(wid)
                b_ov = item.branch_outcomes if item.branch_outcomes is not None else boutcomes.get(wid)
                fp = set(item.forwarded_pcs)
                init_s, init_v = item.initial_sgpr, item.initial_vgpr
                wg_id, w_idx = item.workgroup_id, item.wave_index_in_wg
            else:
                wid, sid, prog = item
                em = masks.get(wid, FULL_EXEC_MASK) & FULL_EXEC_MASK
                v_ov = vstates.get(wid)
                b_ov = boutcomes.get(wid)
                fp = set()
                init_s, init_v = None, None
                wg_id, w_idx = (0, 0, 0), idx

            if not (0 <= sid < self.num_simds):
                raise ValueError(f"simd_id {sid} out of range 0..{self.num_simds - 1}")
            if wid in seen_ids:
                raise ValueError(f"Duplicate wave_id {wid}")
            seen_ids.add(wid)

            insns = _coerce_program(prog, vgpr_state=v_ov, branch_outcomes=b_ov, forwarded_pcs=fp)
            has_decoded = any(ins.decoded is not None for ins in insns)
            if has_decoded and shared_lds is None and len(waves_spec) > 1:
                shared_lds = bytearray(65536)
            isa_state = (
                Wave32ISAState(
                    exec_mask=em, initial_sgpr=init_s, initial_vgpr=init_v,
                    lds_mem=shared_lds, wave_index_in_wg=w_idx, workgroup_id=wg_id,
                )
                if has_decoded else None
            )
            pc_map = {ins.pc: i for i, ins in enumerate(insns)}
            waves.append(_WaveState(
                wave_id=wid, simd_id=sid, program=insns, exec_mask=em,
                isa=isa_state, pc_to_idx=pc_map,
                explicit_vgpr_override=v_ov, explicit_branch_override=b_ov,
            ))

        while True:
            active = [w for w in waves if not w.done()]
            if not active:
                break

            # Advance any leading zero-cycle `s_delay_alu` scheduler hints transparently
            progressed_hint = False
            for w in active:
                while not w.done() and w.current_insn().pipe is Pipe.DELAY_ALU:
                    self._commit_instruction(
                        w, w.current_insn(), issue_cycle=w.next_fetch_cycle,
                        earliest_fetch=w.next_fetch_cycle, alu_src=AluSrc.SALU,
                        dual_issued=False, reserve_simd=False,
                    )
                    progressed_hint = True
            if progressed_hint:
                continue

            for w in active:
                ins = w.current_insn()
                if ins.pipe is Pipe.BARRIER and w.waiting_barrier_id is None:
                    arr = self._compute_ready_cycle(w, ins, w.next_fetch_cycle)
                    w.waiting_barrier_id = ins.barrier_id
                    w.barrier_arrival_cycle = arr

            barrier_ids = {w.waiting_barrier_id for w in active}
            if len(barrier_ids) == 1 and None not in barrier_ids:
                release_cycle = max(w.barrier_arrival_cycle or 0 for w in active)
                for w in sorted(active, key=lambda x: x.wave_id):
                    ins = w.current_insn()
                    earliest = w.next_fetch_cycle
                    issue_cyc = max(release_cycle, self._simd_issue_free[w.simd_id])
                    self._commit_instruction(
                        w, ins, issue_cycle=issue_cyc, earliest_fetch=earliest,
                        alu_src=AluSrc.BARRIER, dual_issued=False, reserve_simd=True,
                    )
                    w.waiting_barrier_id = None
                    w.barrier_arrival_cycle = None
                continue

            runnable = [w for w in active if w.waiting_barrier_id is None]
            if not runnable:
                raise RuntimeError("Workgroup barrier deadlock: waves waiting on mismatched barrier_ids")

            candidates: List[Tuple[int, int, _WaveState]] = []
            for w in runnable:
                ins = w.current_insn()
                ready_t = self._compute_ready_cycle(w, ins, w.next_fetch_cycle)
                candidates.append((ready_t, w.wave_id, w))
            candidates.sort(key=lambda item: (item[0], item[1]))
            issue_t, _, chosen = candidates[0]

            first_ins = chosen.current_insn()
            earliest_first = chosen.next_fetch_cycle

            dual_ok = False
            second_ins: Optional[Instruction] = None
            if chosen.idx + 1 < len(chosen.program):
                cand_second = chosen.program[chosen.idx + 1]
                if cand_second.pipe is not Pipe.DELAY_ALU and can_dual_issue_pair(first_ins, cand_second, self.profile):
                    second_ready = self._compute_ready_cycle(
                        chosen, cand_second, earliest_fetch=issue_t, ignore_simd_slot=True
                    )
                    if second_ready <= issue_t:
                        dual_ok = True
                        second_ins = cand_second

            if dual_ok and second_ins is not None:
                self._commit_instruction(
                    chosen, first_ins, issue_cycle=issue_t, earliest_fetch=earliest_first,
                    alu_src=AluSrc.VALU_SALU, dual_issued=True, reserve_simd=True,
                )
                if not chosen.done() and chosen.current_insn() is second_ins:
                    self._commit_instruction(
                        chosen, second_ins, issue_cycle=issue_t, earliest_fetch=issue_t,
                        alu_src=AluSrc.VALU_SALU, dual_issued=True, reserve_simd=False,
                    )
            else:
                self._commit_instruction(
                    chosen, first_ins, issue_cycle=issue_t, earliest_fetch=earliest_first,
                    alu_src=_default_alu_src(first_ins.pipe, is_vopd=first_ins.is_vopd),
                    dual_issued=False, reserve_simd=True,
                )

        return {w.wave_id: w.events for w in waves}


# ---------------------------------------------------------------------------
# Public Entrypoints: simulate_wave, simulate_cu, trace_mae, fit_profile
# ---------------------------------------------------------------------------

def simulate_wave(
    program: ProgramInput,
    profile: Optional[LatencyProfile] = None,
    contention: Sequence[ContentionEvent] = (),
    initial_exec_mask: int = FULL_EXEC_MASK,
    wave_id: int = 0,
    simd_id: int = 0,
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
    branch_outcomes: Optional[Mapping[int, bool]] = None,
    forwarded_pcs: Optional[Set[int]] = None,
    initial_sgpr: Optional[Mapping[int, int]] = None,
    initial_vgpr: Optional[Mapping[int, Sequence[int]]] = None,
    hw_sqtt_mode: bool = False,
) -> List[CycleEvent]:
    """Simulate a single Wave32 instruction stream (raw GFX11 bytes, ELF, DecodedInsn, or Instruction)."""
    sim = Wave32Simulator(profile=profile, contention=contention, hw_sqtt_mode=hw_sqtt_mode)
    return sim.run_wave(
        program, wave_id=wave_id, simd_id=simd_id,
        initial_exec_mask=initial_exec_mask, vgpr_state=vgpr_state,
        branch_outcomes=branch_outcomes, forwarded_pcs=forwarded_pcs,
        initial_sgpr=initial_sgpr, initial_vgpr=initial_vgpr,
    )


def simulate_cu(
    waves: Sequence[Union[WaveSpec, ProgramInput, Tuple[int, int, ProgramInput]]],
    profile: Optional[LatencyProfile] = None,
    contention: Sequence[ContentionEvent] = (),
    num_simds: int = CU_NUM_SIMDS,
    hw_sqtt_mode: bool = False,
) -> Dict[int, List[CycleEvent]]:
    """Simulate concurrent Wave32 streams on a 4-SIMD32 CU with dynamic shared-port arbitration."""
    sim = Wave32Simulator(profile=profile, contention=contention, num_simds=num_simds, hw_sqtt_mode=hw_sqtt_mode)
    specs: List[Union[WaveSpec, Tuple[int, int, ProgramInput]]] = []
    for idx, item in enumerate(waves):
        if isinstance(item, WaveSpec):
            specs.append(item)
        elif isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], int) and isinstance(item[1], int):
            specs.append(item)
        else:
            specs.append(WaveSpec(program=item, wave_id=idx, simd_id=idx % num_simds, wave_index_in_wg=idx))  # type: ignore[arg-type]
    return sim.run_workgroup(specs)


def _normalize_observed(
    observed: ObservedInput,
    fmt: str = "sqtt",
    wave_id: int = 0,
    program: Optional[ProgramInput] = None,
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    prog_for_align: Optional[Union[bytes, bytearray, memoryview, str, Sequence[DecodedInsn], AMDGPUKernel]] = None
    if isinstance(program, (bytes, bytearray, memoryview, str, AMDGPUKernel)):
        prog_for_align = program
    elif program is not None:
        plist = list(program)
        if plist and isinstance(plist[0], DecodedInsn):
            prog_for_align = plist  # type: ignore[assignment]

    if isinstance(observed, (bytes, bytearray, memoryview, str)):
        by_wave = sqtt_to_tuples(observed, program=prog_for_align)
        chosen = by_wave.get(wave_id) or (by_wave[sorted(by_wave.keys())[0]] if by_wave else [])
        return chosen, "sqtt_hw"

    obs_list = list(observed)
    if not obs_list:
        return [], fmt
    if isinstance(obs_list[0], SQTTAlignedEvent):
        return [e.to_sqtt_tuple() for e in obs_list], "sqtt_hw"  # type: ignore[union-attr]
    if isinstance(obs_list[0], SQTTPacket):
        by_wave = sqtt_to_tuples(obs_list, program=prog_for_align)  # type: ignore[arg-type]
        chosen = by_wave.get(wave_id) or (by_wave[sorted(by_wave.keys())[0]] if by_wave else [])
        return chosen, "sqtt_hw"
    if isinstance(obs_list[0], CycleEvent):
        if fmt == "sqtt":
            return [e.to_sqtt() for e in obs_list], "sqtt"  # type: ignore[union-attr]
        return [e.to_tuple() for e in obs_list], "tuple"  # type: ignore[union-attr]
    return [tuple(x) for x in obs_list], fmt  # type: ignore[misc]


def trace_mae(
    predicted: Sequence[CycleEvent],
    observed: Sequence[Tuple[int, int, int, int]],
    fmt: str = "sqtt",
) -> float:
    """Compute Mean Absolute Error between predicted CycleEvents and observed tuples.

    For `fmt="sqtt_hw"` (real hardware SQTT v2 issue timestamps), computes MAE over
    exact hardware issue cycles `|pred.issue_cycle - obs_time|`.
    For `fmt="sqtt"` / `"tuple"`, computes MAE over all 3 timing columns.
    """
    if len(predicted) != len(observed):
        raise ValueError(f"Trace length mismatch: predicted {len(predicted)} vs observed {len(observed)}")
    if not predicted:
        return 0.0
    if fmt == "sqtt_hw":
        total = 0.0
        for ev, obs in zip(predicted, observed):
            if ev.pc != obs[0]:
                raise ValueError(f"PC mismatch: predicted 0x{ev.pc:x} vs observed 0x{obs[0]:x}")
            total += abs(ev.issue_cycle - obs[3])
        return total / len(predicted)

    total_err = 0.0
    for ev, obs in zip(predicted, observed):
        pred_t = ev.to_sqtt() if fmt == "sqtt" else ev.to_tuple()
        if pred_t[0] != obs[0]:
            raise ValueError(f"PC mismatch: predicted {pred_t[0]} vs observed {obs[0]}")
        total_err += abs(pred_t[1] - obs[1]) + abs(pred_t[2] - obs[2]) + abs(pred_t[3] - obs[3])
    return total_err / (3.0 * len(predicted))


def _evaluate_profile_metrics(
    profile: LatencyProfile,
    prepared: Sequence[Tuple[
        ProgramInput, List[Tuple[int, int, int, int]], str, Tuple[ContentionEvent, ...],
        int, int, int, Optional[Mapping[str, Sequence[int]]], Optional[Mapping[int, bool]],
        Optional[Mapping[int, int]], Optional[Mapping[int, Sequence[int]]], bool
    ]],
    delta_only: bool = False,
) -> Tuple[float, float]:
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0
    for prog, obs, fmt, cont, mask, wid, sid, vstate, bout, init_s, init_v, hw_mode in prepared:
        pred = simulate_wave(
            prog, profile=profile, contention=cont,
            initial_exec_mask=mask, wave_id=wid, simd_id=sid,
            vgpr_state=vstate, branch_outcomes=bout,
            initial_sgpr=init_s, initial_vgpr=init_v,
            hw_sqtt_mode=hw_mode,
        )
        if len(pred) != len(obs):
            raise ValueError(f"Trace length mismatch: predicted {len(pred)} vs observed {len(obs)}")
        if fmt == "sqtt_hw":
            if delta_only:
                for idx in range(1, len(pred)):
                    if pred[idx].pc != obs[idx][0]:
                        raise ValueError(f"PC mismatch: predicted 0x{pred[idx].pc:x} vs observed 0x{obs[idx][0]:x}")
                    p_dt = pred[idx].issue_cycle - pred[idx - 1].issue_cycle
                    o_dt = obs[idx][3] - obs[idx - 1][3]
                    diff = p_dt - o_dt
                    total_abs += abs(diff)
                    total_sq += float(diff * diff)
                    total_n += 1
            else:
                for ev, o in zip(pred, obs):
                    if ev.pc != o[0]:
                        raise ValueError(f"PC mismatch: predicted 0x{ev.pc:x} vs observed 0x{o[0]:x}")
                    diff = ev.issue_cycle - o[3]
                    total_abs += abs(diff)
                    total_sq += float(diff * diff)
                    total_n += 1
        else:
            for ev, o in zip(pred, obs):
                pt = ev.to_sqtt() if fmt == "sqtt" else ev.to_tuple()
                if pt[0] != o[0]:
                    raise ValueError(f"PC mismatch: predicted {pt[0]} vs observed {o[0]}")
                for k in (1, 2, 3):
                    diff = pt[k] - o[k]
                    total_abs += abs(diff)
                    total_sq += float(diff * diff)
                total_n += 3
    if total_n == 0:
        return 0.0, 0.0
    return total_sq, total_abs / total_n


def _evaluate_profile_mae(profile: LatencyProfile, traces: Sequence[TraceSample]) -> float:
    prepared = _prepare_trace_samples(traces)
    _, mae = _evaluate_profile_metrics(profile, prepared)
    return mae


def _prepare_trace_samples(traces: Sequence[TraceSample]):
    prepared = []
    for s in traces:
        obs, fmt = _normalize_observed(s.observed, fmt=s.resolved_format(), wave_id=s.wave_id, program=s.program)
        insns = _coerce_program(s.program, vgpr_state=s.vgpr_state, branch_outcomes=s.branch_outcomes)
        hw_mode = s.hw_sqtt_mode if s.hw_sqtt_mode is not None else (fmt == "sqtt_hw")
        if s.hw_sqtt_mode is None and fmt == "sqtt_hw":
            non_delay = [i for i in insns if i.pipe is not Pipe.DELAY_ALU]
            if len(non_delay) == len(obs) and non_delay and non_delay[-1].pipe is Pipe.ENDPGM:
                hw_mode = False
        prepared.append((
            insns, obs, fmt, s.contention,
            s.initial_exec_mask, s.wave_id, s.simd_id,
            s.vgpr_state, s.branch_outcomes, s.initial_sgpr, s.initial_vgpr, hw_mode,
        ))
    return prepared


def fit_profile(
    traces: Sequence[TraceSample],
    initial: Optional[LatencyProfile] = None,
    initial_profile: Optional[LatencyProfile] = None,
    max_rounds: int = 8,
) -> Tuple[LatencyProfile, float]:
    """Calibrate LatencyProfile via structural init + multi-scale least-squares/coordinate descent.

    Works on BOTH synthetic `CycleEvent`/tuple traces AND real AMD GFX1100 hardware
    ELF + SQTT v2 traces (optimizing exact hardware issue-cycle error with zero circularity).
    """
    base = initial_profile or initial or LatencyProfile()
    if not traces:
        return base, 0.0

    prepared = _prepare_trace_samples(traces)
    has_hw_sqtt = any(item[2] == "sqtt_hw" for item in prepared)

    # Structural warm-start directly from observed timing gaps / durations
    init_kwargs: Dict[str, object] = {}
    for prog, obs, fmt, cont, mask, _wid, _sid, vstate, bout, _is, _iv, hw_mode in prepared:
        insns = [
            i for i in prog  # type: ignore[union-attr]
            if i.pipe is not Pipe.DELAY_ALU and not (hw_mode and i.pipe is Pipe.ENDPGM)
        ]
        if fmt == "sqtt_hw" and len(insns) == len(obs):
            # Infer cold SMEM/VMEM/LDS wait latencies from observed hardware timestamp jumps
            last_smem_t: Optional[int] = None
            last_vmem_t: Optional[int] = None
            last_lds_t: Optional[int] = None
            last_lds_is_wr: bool = False
            last_valu_t: Optional[int] = None
            for idx, (ins, o) in enumerate(zip(insns, obs)):
                t_obs = o[3]
                if ins.pipe is Pipe.VALU32:
                    last_valu_t = t_obs
                elif ins.pipe is Pipe.SMEM_RD:
                    last_smem_t = t_obs
                elif ins.pipe in VMEM_PIPES:
                    if last_valu_t is not None and 10 <= (t_obs - last_valu_t) <= 24:
                        init_kwargs.setdefault("valu_to_mem_latency", t_obs - last_valu_t)
                    if ins.pipe is Pipe.VMEM_RD:
                        last_vmem_t = t_obs
                elif ins.pipe in LDS_PIPES:
                    if last_valu_t is not None and 10 <= (t_obs - last_valu_t) <= 24:
                        init_kwargs.setdefault("valu_to_mem_latency", t_obs - last_valu_t)
                    last_lds_t = t_obs
                    last_lds_is_wr = ins.pipe in LDS_WR_PIPES
                elif ins.pipe is Pipe.WAIT_LGKMCNT:
                    if ins.lgkm_target < WAITCNT_LGKM_NONE:
                        if last_smem_t is not None and (last_lds_t is None or last_smem_t > last_lds_t):
                            est = max(1, t_obs - last_smem_t)
                            if est > 25:
                                init_kwargs["smem_rd_latency"] = min(2000, est)
                            last_smem_t = None
                        elif last_lds_t is not None:
                            est = max(1, t_obs - last_lds_t)
                            if est > 5:
                                if last_lds_is_wr:
                                    init_kwargs["lds_wr_latency_overhead"] = min(96, max(0, est - 2))
                                else:
                                    init_kwargs["lds_rd_latency"] = min(160, est)
                            last_lds_t = None
                    if ins.vm_target < WAITCNT_VM_NONE and last_vmem_t is not None:
                        est = max(1, t_obs - last_vmem_t)
                        if est > 25:
                            init_kwargs["vmem_rd_latency"] = min(2500, est)
                        last_vmem_t = None
        elif fmt != "sqtt_hw":
            for ins, o in zip(insns, obs):
                dur = o[2] if fmt == "sqtt" else o[3]
                if dur <= 0:
                    continue
                if ins.pipe is Pipe.SALU:
                    init_kwargs.setdefault("salu_latency", dur); init_kwargs.setdefault("salu_issue", dur)
                elif ins.pipe is Pipe.SALU_WR_EXEC:
                    init_kwargs.setdefault("salu_wr_exec_latency", dur); init_kwargs.setdefault("salu_wr_exec_issue", dur)
                elif ins.pipe is Pipe.VALU32:
                    init_kwargs.setdefault("valu32_latency", dur)
                elif ins.pipe is Pipe.VALUT_4:
                    init_kwargs.setdefault("valut_4_latency", dur)
                elif ins.pipe is Pipe.VALUB_2:
                    init_kwargs.setdefault("valub_2_latency", dur); init_kwargs.setdefault("valub_2_issue", dur)
                elif ins.pipe is Pipe.VALUB_4:
                    init_kwargs.setdefault("valub_4_latency", dur); init_kwargs.setdefault("valub_4_issue", dur)
                elif ins.pipe is Pipe.VALUB_16:
                    init_kwargs.setdefault("valub_16_latency", dur); init_kwargs.setdefault("valub_16_issue", dur)
                elif ins.pipe is Pipe.SMEM_RD:
                    init_kwargs.setdefault("smem_rd_latency", dur)
                elif ins.pipe is Pipe.VMEM_RD and not cont:
                    init_kwargs.setdefault("vmem_rd_latency", dur)
                elif ins.pipe is Pipe.VMEM_WR and not cont:
                    init_kwargs.setdefault("vmem_wr_latency", dur)
                elif ins.pipe is Pipe.LDS_RD and not cont:
                    info = analyze_lds_access(ins.pipe, ins.lds_addrs, mask, base, ins.lds_dwords, ins.lds_sub_offsets)
                    est = dur - info.conflict_extra
                    if est >= 1:
                        init_kwargs.setdefault("lds_rd_latency", est)
                elif ins.pipe in LDS_WR_PIPES and not cont:
                    info = analyze_lds_access(ins.pipe, ins.lds_addrs, mask, base, ins.lds_dwords, ins.lds_sub_offsets)
                    est = dur - info.port_cycles
                    if est >= 0:
                        init_kwargs.setdefault("lds_wr_latency_overhead", est)
                elif ins.pipe is Pipe.JUMP:
                    init_kwargs.setdefault("jump_taken_cycles" if ins.taken else "jump_untaken_cycles", dur)
                elif ins.pipe is Pipe.BARRIER:
                    init_kwargs.setdefault("barrier_cycles", dur)
                elif ins.pipe is Pipe.WAIT_LGKMCNT:
                    init_kwargs.setdefault("waitcnt_cycles", dur)

    best_prof = base
    best_sse, best_mae = _evaluate_profile_metrics(best_prof, prepared)
    if best_mae == 0.0:
        return best_prof, 0.0

    if init_kwargs:
        warm = dataclasses.replace(base, **init_kwargs)
        w_sse, w_mae = _evaluate_profile_metrics(warm, prepared)
        if (w_sse, w_mae) < (best_sse, best_mae):
            best_prof, best_sse, best_mae = warm, w_sse, w_mae
            if best_mae == 0.0:
                return best_prof, 0.0

    # Grid over dual-issue & VALU32 bypass forwarding
    for dual_flag in (True, False):
        for fwd_flag in (False, True):
            for fwd_lat in ((1, 2) if fwd_flag else (best_prof.valu32_forward_latency,)):
                cand = dataclasses.replace(
                    best_prof, enable_dual_issue=dual_flag,
                    valu32_forwarding=fwd_flag, valu32_forward_latency=fwd_lat,
                )
                sse, mae = _evaluate_profile_metrics(cand, prepared)
                if (sse, mae) < (best_sse, best_mae):
                    best_prof, best_sse, best_mae = cand, sse, mae
                    if best_mae == 0.0:
                        return best_prof, 0.0

    present_pipes: Set[Pipe] = set()
    for insns_item in prepared:
        for ins in insns_item[0]:  # type: ignore[union-attr]
            present_pipes.add(ins.pipe)

    all_bounds: List[Tuple[str, Tuple[int, int], Set[Pipe]]] = []
    if has_hw_sqtt and ("valu_to_mem_latency" in init_kwargs or best_prof.valu_to_mem_latency > 0 or best_prof.salu_to_mem_latency > 0):
        all_bounds.extend([
            ("valu_to_mem_latency", (0, 32), LDS_PIPES | VMEM_PIPES),
            ("salu_to_mem_latency", (0, 24), LDS_PIPES | VMEM_PIPES),
            ("waitcnt_resume_gap", (0, 8), {Pipe.WAIT_LGKMCNT}),
        ])
    all_bounds.extend([
        ("lds_rd_base", (1, 16), {Pipe.LDS_RD}),
        ("valu32_issue", (1, 8), {Pipe.VALU32}),
        ("valu32_forward_latency", (1, 8), {Pipe.VALU32}),
        ("valut_4_issue", (1, 16), {Pipe.VALUT_4}),
        ("valub_2_issue", (1, 16), {Pipe.VALUB_2}),
        ("valub_4_issue", (1, 16), {Pipe.VALUB_4}),
        ("valub_16_issue", (1, 32), {Pipe.VALUB_16}),
        ("salu_issue", (1, 8), {Pipe.SALU}),
        ("salu_wr_exec_issue", (1, 8), {Pipe.SALU_WR_EXEC}),
        ("smem_rd_issue", (1, 16), {Pipe.SMEM_RD}),
        ("vmem_rd_issue", (1, 16), {Pipe.VMEM_RD}),
        ("vmem_wr_issue", (1, 16), {Pipe.VMEM_WR}),
        ("valu32_latency", (1, 32), {Pipe.VALU32}),
        ("valu_wr_exec_latency", (1, 32), {Pipe.VALU1_WR_EXEC}),
        ("valut_4_latency", (1, 64), {Pipe.VALUT_4}),
        ("valub_2_latency", (1, 32), {Pipe.VALUB_2}),
        ("valub_4_latency", (1, 32), {Pipe.VALUB_4}),
        ("valub_16_latency", (1, 64), {Pipe.VALUB_16}),
        ("salu_latency", (1, 8), {Pipe.SALU}),
        ("salu_wr_exec_latency", (1, 16), {Pipe.SALU_WR_EXEC}),
        ("jump_taken_cycles", (1, 24), {Pipe.JUMP}),
        ("jump_untaken_cycles", (1, 16), {Pipe.JUMP}),
        ("barrier_cycles", (1, 64), {Pipe.BARRIER}),
        ("waitcnt_cycles", (1, 16), {Pipe.WAIT_LGKMCNT}),
        ("lds_rd_latency", (1, 160), {Pipe.LDS_RD}),
        ("lds_wr_latency_overhead", (0, 96), set(LDS_WR_PIPES.keys())),
        ("smem_rd_latency", (1, 2000), {Pipe.SMEM_RD}),
        ("vmem_rd_latency", (1, 2500), {Pipe.VMEM_RD}),
        ("vmem_wr_latency", (1, 400), {Pipe.VMEM_WR}),
    ])
    tunable_bounds: Dict[str, Tuple[int, int]] = {
        fname: bnd for fname, bnd, req in all_bounds if (present_pipes & req)
    }

    coupled_pairs = (
        ("lds_rd_base", "lds_rd_latency"), ("valu32_issue", "valu32_latency"),
        ("valut_4_issue", "valut_4_latency"), ("vmem_rd_issue", "vmem_rd_latency"),
        ("waitcnt_cycles", "smem_rd_latency"), ("waitcnt_cycles", "lds_rd_latency"),
    )

    # Stage 1 (real hardware SQTT): decoupled inter-instruction delta-time calibration
    # eliminates cumulative timestamp coupling so every pipe latency snaps cleanly onto its
    # true hardware value before Stage 2 cumulative issue-time refinement.
    if has_hw_sqtt:
        d_sse, d_mae = _evaluate_profile_metrics(best_prof, prepared, delta_only=True)
        for _ in range(2):
            d_imp = False
            for bfield in ("enable_dual_issue", "valu32_forwarding"):
                flipped = dataclasses.replace(best_prof, **{bfield: not getattr(best_prof, bfield)})
                sse, mae = _evaluate_profile_metrics(flipped, prepared, delta_only=True)
                if (sse, mae) < (d_sse, d_mae):
                    best_prof, d_sse, d_mae = flipped, sse, mae
                    d_imp = True
            for fname, (lo, hi) in tunable_bounds.items():
                cur = int(getattr(best_prof, fname))
                steps = (256, 128, 64, 32, 16, 8, 4, 2, 1) if hi >= 400 else (32, 16, 8, 4, 2, 1)
                for step in steps:
                    if step > (hi - lo):
                        continue
                    for delta in (-step, step):
                        val = cur + delta
                        if val < lo or val > hi:
                            continue
                        cand = dataclasses.replace(best_prof, **{fname: val})
                        sse, mae = _evaluate_profile_metrics(cand, prepared, delta_only=True)
                        if (sse, mae) < (d_sse, d_mae):
                            best_prof, d_sse, d_mae = cand, sse, mae
                            cur = val
                            d_imp = True
            if not d_imp or d_mae == 0.0:
                break
        best_sse, best_mae = _evaluate_profile_metrics(best_prof, prepared)
        if best_mae == 0.0:
            return best_prof, 0.0

    for _ in range(max_rounds):
        improved = False
        for bfield in ("enable_dual_issue", "valu32_forwarding"):
            flipped = dataclasses.replace(best_prof, **{bfield: not getattr(best_prof, bfield)})
            sse, mae = _evaluate_profile_metrics(flipped, prepared)
            if (sse, mae) < (best_sse, best_mae):
                best_prof, best_sse, best_mae = flipped, sse, mae
                improved = True
                if best_mae == 0.0:
                    return best_prof, 0.0

        for fname, (lo, hi) in tunable_bounds.items():
            cur = int(getattr(best_prof, fname))
            steps = (256, 128, 64, 32, 16, 8, 4, 2, 1) if hi >= 400 else (32, 16, 8, 4, 2, 1)
            for step in steps:
                if step > (hi - lo):
                    continue
                for delta in (-step, step):
                    val = cur + delta
                    if val < lo or val > hi:
                        continue
                    cand = dataclasses.replace(best_prof, **{fname: val})
                    sse, mae = _evaluate_profile_metrics(cand, prepared)
                    if (sse, mae) < (best_sse, best_mae):
                        best_prof, best_sse, best_mae = cand, sse, mae
                        cur = val
                        improved = True
                        if best_mae == 0.0:
                            return best_prof, 0.0

        if not improved and best_mae > 0.0:
            if "valu32_latency" in tunable_bounds and "valu32_forward_latency" in tunable_bounds:
                cur_lat = best_prof.valu32_latency
                for fwd_flag in (True, False):
                    for fwd_lat in ((1, 2, 3) if fwd_flag else (best_prof.valu32_forward_latency,)):
                        for dlat in (-3, -2, -1, 0, 1, 2, 3, 4):
                            vlat = cur_lat + dlat
                            if not (1 <= vlat <= 32):
                                continue
                            cand = dataclasses.replace(
                                best_prof, valu32_forwarding=fwd_flag,
                                valu32_forward_latency=fwd_lat, valu32_latency=vlat,
                            )
                            sse, mae = _evaluate_profile_metrics(cand, prepared)
                            if (sse, mae) < (best_sse, best_mae):
                                best_prof, best_sse, best_mae = cand, sse, mae
                                improved = True
                                if best_mae == 0.0:
                                    return best_prof, 0.0
            for f1, f2 in coupled_pairs:
                if f1 not in tunable_bounds or f2 not in tunable_bounds:
                    continue
                lo1, hi1 = tunable_bounds[f1]
                lo2, hi2 = tunable_bounds[f2]
                c1, c2 = int(getattr(best_prof, f1)), int(getattr(best_prof, f2))
                for d1 in (-2, -1, 0, 1, 2):
                    for d2 in (-2, -1, 0, 1, 2):
                        if d1 == 0 and d2 == 0:
                            continue
                        v1, v2 = c1 + d1, c2 + d2
                        if lo1 <= v1 <= hi1 and lo2 <= v2 <= hi2:
                            cand = dataclasses.replace(best_prof, **{f1: v1, f2: v2})
                            sse, mae = _evaluate_profile_metrics(cand, prepared)
                            if (sse, mae) < (best_sse, best_mae):
                                best_prof, best_sse, best_mae = cand, sse, mae
                                improved = True
                                if best_mae == 0.0:
                                    return best_prof, 0.0
        if not improved:
            break

    return best_prof, best_mae


# ---------------------------------------------------------------------------
# Real Hardware AMD GFX1100 (RX 7900 XTX) ELF + SQTT v2 Validation Pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareValidationReport:
    """Validation metrics comparing Wave32Simulator against a real GFX1100 hardware SQTT v2 capture."""

    kernel_name: str
    wave_id: int
    simd_id: int
    num_insts: int
    total_hw_cycles: int
    total_pred_cycles: int
    issue_time_mae: float
    delta_mae: float
    max_abs_err: int
    relative_err_pct: float
    profile: LatencyProfile
    aligned_events: Tuple[SQTTAlignedEvent, ...]
    predicted_events: Tuple[CycleEvent, ...]


def validate_hardware_sqtt(
    elf_bytes: Union[bytes, bytearray, memoryview],
    sqtt_bytes: Union[bytes, bytearray, memoryview, str],
    *,
    profile: Optional[LatencyProfile] = None,
    fit: bool = True,
    wave_id: Optional[int] = None,
    include_contention: bool = False,
) -> HardwareValidationReport:
    """Validate simulator predictions against a real AMD GFX1100 AMDGPU ELF + SQTT v2 capture."""
    kern, decoded = decode_elf(elf_bytes)
    pkts = parse_sqtt(sqtt_bytes)
    aligned_map = align_sqtt_to_program(pkts, decoded)
    if not aligned_map:
        raise ValueError(f"No traced instruction events found in SQTT blob for {kern.name}")

    chosen_key: Optional[Tuple[int, int]] = None
    if wave_id is not None:
        for k in sorted(aligned_map.keys()):
            if k[1] == wave_id:
                chosen_key = k
                break
    if chosen_key is None:
        chosen_key = sorted(aligned_map.keys())[0]

    hw_evs = aligned_map[chosen_key]
    simd_id, wid = chosen_key
    t0 = hw_evs[0].timestamp
    cont = tuple(extract_sqtt_contention(pkts, base_timestamp=t0)) if include_contention else ()

    sample = TraceSample(
        program=bytes(elf_bytes),
        observed=hw_evs,
        contention=cont,
        wave_id=wid,
        simd_id=simd_id,
        format="sqtt_hw",
        hw_sqtt_mode=True,
    )
    if fit:
        prof, issue_mae = fit_profile([sample], initial=profile)
    else:
        prof = profile or LatencyProfile()
        issue_mae = _evaluate_profile_mae(prof, [sample])

    pred = simulate_wave(
        bytes(elf_bytes), profile=prof, contention=cont,
        wave_id=wid, simd_id=simd_id, hw_sqtt_mode=True,
    )
    abs_errs = [abs(p.issue_cycle - h.issue_time) for p, h in zip(pred, hw_evs)]
    dt_errs = [
        abs((pred[i].issue_cycle - pred[i - 1].issue_cycle) - (hw_evs[i].issue_time - hw_evs[i - 1].issue_time))
        for i in range(1, len(pred))
    ]
    total_hw = hw_evs[-1].issue_time if hw_evs else 0
    total_pred = pred[-1].issue_cycle if pred else 0
    rel_pct = (100.0 * issue_mae / total_hw) if total_hw > 0 else 0.0
    return HardwareValidationReport(
        kernel_name=kern.name,
        wave_id=wid,
        simd_id=simd_id,
        num_insts=len(hw_evs),
        total_hw_cycles=total_hw,
        total_pred_cycles=total_pred,
        issue_time_mae=issue_mae,
        delta_mae=(sum(dt_errs) / len(dt_errs)) if dt_errs else 0.0,
        max_abs_err=max(abs_errs) if abs_errs else 0,
        relative_err_pct=rel_pct,
        profile=prof,
        aligned_events=tuple(hw_evs),
        predicted_events=tuple(pred),
    )

# ---------------------------------------------------------------------------
# Unmodified Real Hardware AMD Radeon RX 7900 XTX (GFX1100) ELF + SQTT v2 Fixtures
# Source: tinygrad/test/amd/sqtt/examples/gfx1100/ (AMDGPU KFD / ROCprofv3 captures)
# ---------------------------------------------------------------------------

_REAL_GFX1100_HW_FIXTURES: Dict[str, Dict[str, object]] = {
    "E_3": {
        "source": "tinygrad/test/amd/sqtt/examples/gfx1100/profile_plus_run_0.pkl",
        "se": 1,
        "elf_len": 4152,
        "elf_sha256": "def5241fda3343ccc7a64942c7b342b3fe7d7eea57be0dadd82b7646f2f414ea",
        "elf_zb64": (
            "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"
            "cQDYHmL/N1oN3O4="
        ),
        "sqtt_len": 4672,
        "sqtt_sha256": "b758acb0da91a5f3cb92f010c0f8f3721c77d910b0c1db424cb2d2859c50e38c",
        "sqtt_zb64": (
            "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"
            "UctPoJbXEv1roo+GRfnfHWQkXQ=="
        ),
    },
    "custom_lds_sync": {
        "source": "tinygrad/test/amd/sqtt/examples/gfx1100/profile_sync_run_0.pkl",
        "se": 1,
        "elf_len": 1367,
        "elf_sha256": "f2c2ba3250744b1d5dd0aeefcc20aa7c29f75855ac66f0668c986a49523d4558",
        "elf_zb64": (
            "eNqrd/VxY2IgDKazYIoxMxCjc7ADli9A4gcDQ+f+JgYmAwYGkxuMYPHO/QwMe/e3MLF4/WeYUVMPFrtxgwnkbaDc"
            "QSaOOhZ2LqYmBh6gnqw7bKwg+Q1APfNH8SgewphEAC0COKDcD3/+gwFuDWscWkDqYeWJXklqRQmDXlF+SmJJIoNe"
            "cUlRSWISAygLgjAbTBcjKi3AjN9V7FClsALKgQWNJuArfnDxhgkaoPrFCegHAJUHGAg="
        ),
        "sqtt_len": 4704,
        "sqtt_sha256": "6bf6a7182ad589a86db598652290f8855194b7db9201afdb45aab9aeced51d8a",
        "sqtt_zb64": (
            "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"
            "hxE5UQ=="
        ),
    },
    "r_32_32_16_2": {
        "source": "tinygrad/test/amd/sqtt/examples/gfx1100/profile_gemm_run_0.pkl",
        "se": 0,
        "elf_len": 4752,
        "elf_sha256": "2663451b95c7ffecd4806d9eb4f3696293d05b7eec85da399f8acf848eaa7bcf",
        "elf_zb64": (
            "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"
        ),
        "sqtt_len": 6976,
        "sqtt_sha256": "56b658d2bc946a4f6f01093422c48cdf4bf61ec6822398a750e116d00f732ba7",
        "sqtt_zb64": (
            "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"
        ),
    },
}

def load_real_gfx1100_fixture(name: str) -> Tuple[bytes, bytes]:
    """Return unmodified real RX 7900 XTX (gfx1100) `(elf_bytes, sqtt_bytes)` verified by SHA-256."""
    fx = _REAL_GFX1100_HW_FIXTURES[name]
    elf = zlib.decompress(base64.b64decode(str(fx["elf_zb64"])))
    sqtt = zlib.decompress(base64.b64decode(str(fx["sqtt_zb64"])))
    if hashlib.sha256(elf).hexdigest() != fx["elf_sha256"]:
        raise RuntimeError(f"Fixture {name} ELF SHA-256 mismatch")
    if hashlib.sha256(sqtt).hexdigest() != fx["sqtt_sha256"]:
        raise RuntimeError(f"Fixture {name} SQTT SHA-256 mismatch")
    return elf, sqtt

class TestWave32Cycles(unittest.TestCase):
    """Comprehensive deterministic verification suite for Wave32 cycle predictor."""

    def test_01_salu_sequential_issue_and_latency(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s2",), src=("s0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s3",), src=("s2",)),
            Instruction(pc=0x08, pipe=Pipe.SALU, dst=("s4",), src=("s3",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual([e.to_tuple() for e in ev], [
            (0x00, 0, 0, 1),
            (0x04, 1, 0, 1),
            (0x08, 2, 0, 1),
        ])

    def test_02_salu_wr_exec_two_cycle_occupancy_and_raw(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SALU_WR_EXEC, exec_mask=0x0000FFFF),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
        ]
        ev = simulate_wave(prog)
        # SALU_WR_EXEC takes 2 cycles and writes EXEC; VALU implicitly reads EXEC => stalls 1 cyc
        self.assertEqual(ev[0].to_tuple(), (0x00, 0, 0, 2))
        self.assertEqual(ev[1].to_tuple(), (0x04, 2, 1, 4))
        self.assertFalse(ev[1].dual_issued)

    def test_03_valu_salu_dual_issue_no_hazard(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s4",), src=("s2",)),
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v2",), src=("v3",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 0)
        self.assertEqual(ev[0].alu_src, AluSrc.VALU_SALU)
        self.assertEqual(ev[1].alu_src, AluSrc.VALU_SALU)
        self.assertTrue(ev[0].dual_issued and ev[1].dual_issued)
        self.assertEqual(ev[1].stall_cycles, 0)
        # Third instruction cannot triple-issue in cycle 0; issues in cycle 1
        self.assertEqual(ev[2].to_tuple(), (0x08, 1, 0, 4))

    def test_04_salu_valu_dual_issue_reverse_order(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s4",), src=("s2",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 0)
        self.assertEqual(ev[1].alu_src, AluSrc.VALU_SALU)

    def test_05_dual_issue_blocked_by_raw_hazard(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s4",), src=("s2",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v1",), src=("v0", "s4")),
        ]
        ev = simulate_wave(prog)
        self.assertFalse(ev[0].dual_issued)
        self.assertFalse(ev[1].dual_issued)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 1)

    def test_06_dual_issue_blocked_by_waw_hazard(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("vcc",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("vcc",), src=("s2",)),
        ]
        ev = simulate_wave(prog)
        self.assertFalse(ev[0].dual_issued)
        self.assertFalse(ev[1].dual_issued)
        # WAW protection: VALU32 completes at t=4, SALU has dur=1 => SALU waits until t=3
        self.assertEqual(ev[0].completion_cycle, 4)
        self.assertEqual(ev[1].issue_cycle, 3)
        self.assertEqual(ev[1].completion_cycle, 4)

    def test_07_dual_issue_blocked_same_pipe_valu_valu(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v2",), src=("v3",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 1)
        self.assertFalse(ev[1].dual_issued)

    def test_08_dual_issue_blocked_when_second_stalled_on_earlier_raw(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s5",), src=("s2",)),
            # Independent SALU followed by VALU depending on v1 (ready at t=4)
            Instruction(pc=0x08, pipe=Pipe.SALU, dst=("s6",), src=("s5",)),
            Instruction(pc=0x0C, pipe=Pipe.VALU32, dst=("v2",), src=("v1",)),
        ]
        ev = simulate_wave(prog)
        # (0x00, 0x04) co-issue at t=0
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 0)
        # 0x08 issues at t=1; 0x0C needs v1 at t=4 so CANNOT co-issue at t=1
        self.assertEqual(ev[2].issue_cycle, 1)
        self.assertFalse(ev[2].dual_issued)
        self.assertEqual(ev[3].issue_cycle, 4)
        self.assertEqual(ev[3].stall_cycles, 2)

    def test_09_valu32_raw_stall_without_forwarding(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v2",), src=("v1",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].to_tuple(), (0x00, 0, 0, 4))
        self.assertEqual(ev[1].to_tuple(), (0x04, 4, 3, 4))

    def test_10_valu32_raw_forwarding_eliminates_stall(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v2",), src=("v1",), forwarded=True),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].to_tuple(), (0x00, 0, 0, 4))
        self.assertEqual(ev[1].to_tuple(), (0x04, 1, 0, 4))

    def test_11_valu32_independent_hides_latency(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v3",), src=("v2",)),
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v5",), src=("v4",)),
            Instruction(pc=0x0C, pipe=Pipe.VALU32, dst=("v7",), src=("v6",)),
            Instruction(pc=0x10, pipe=Pipe.VALU32, dst=("v8",), src=("v1",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual([e.issue_cycle for e in ev], [0, 1, 2, 3, 4])
        self.assertEqual([e.stall_cycles for e in ev], [0, 0, 0, 0, 0])

    def test_12_valut_4_issue_occupancy_and_16_cycle_latency(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALUT_4, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v3",), src=("v2",)),  # blocked by 4-cyc VALU occupancy
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v4",), src=("v1",)),  # blocked until t=16
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].to_tuple(), (0x00, 0, 0, 16))
        self.assertEqual(ev[1].to_tuple(), (0x04, 4, 3, 4))
        self.assertEqual(ev[2].to_tuple(), (0x08, 16, 11, 4))

    def test_13_valub_2_4_16_pipes(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALUB_2, dst=("v2",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.VALUB_4, dst=("v4",), src=("v2",)),
            Instruction(pc=0x08, pipe=Pipe.VALUB_16, dst=("v6",), src=("v4",)),
            Instruction(pc=0x0C, pipe=Pipe.VALU32, dst=("v8",), src=("v6",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual([e.to_tuple() for e in ev], [
            (0x00, 0, 0, 2),
            (0x04, 2, 1, 4),
            (0x08, 6, 3, 16),
            (0x0C, 22, 15, 4),
        ])

    def test_14_jump_taken_vs_untaken(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.JUMP, taken=False),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s1",), src=("s0",)),
            Instruction(pc=0x08, pipe=Pipe.JUMP, taken=True),
            Instruction(pc=0x40, pipe=Pipe.SALU, dst=("s2",), src=("s1",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual([e.to_tuple() for e in ev], [
            (0x00, 0, 0, 1),
            (0x04, 1, 0, 1),
            (0x08, 2, 0, 4),
            (0x40, 6, 0, 1),
        ])

    def test_15_lds_zero_conflict_unit_stride(self) -> None:
        addrs = tuple(lane * 4 for lane in range(WAVE_LANES))
        info = analyze_lds_access(Pipe.LDS_RD, addrs)
        self.assertEqual((info.half0_max_way, info.half1_max_way), (1, 1))
        self.assertEqual(info.conflict_extra, 0)
        self.assertEqual(info.port_cycles, 2)

    def test_16_lds_broadcast_zero_conflict(self) -> None:
        # All 32 lanes access the exact same dword address -> 1 phase per half-wave, 0 conflict!
        addrs = (0x100,) * WAVE_LANES
        info = analyze_lds_access(Pipe.LDS_RD, addrs)
        self.assertEqual((info.half0_max_way, info.half1_max_way), (1, 1))
        self.assertEqual(info.conflict_extra, 0)
        self.assertEqual(info.port_cycles, 2)

    def test_17_lds_sub_dword_same_dword_broadcast(self) -> None:
        # Distinct byte offsets within the SAME 4-byte dword share the same dword address -> 0 conflict
        addrs = tuple(0x80 + (lane % 4) for lane in range(WAVE_LANES))
        info = analyze_lds_access(Pipe.LDS_RD, addrs)
        self.assertEqual(info.conflict_extra, 0)
        self.assertEqual(info.port_cycles, 2)

    def test_18_lds_16_way_bank_conflict_both_halves(self) -> None:
        # Stride 128 bytes (32 dwords) maps every lane to bank 0 with 16 distinct dwords per half-wave
        addrs = tuple(lane * 128 for lane in range(WAVE_LANES))
        info = analyze_lds_access(Pipe.LDS_RD, addrs)
        self.assertEqual((info.half0_max_way, info.half1_max_way), (16, 16))
        self.assertEqual(info.conflict_extra, 30)
        self.assertEqual(info.port_cycles, 32)

    def test_19_lds_asymmetric_half_wave_conflict_and_partial_broadcast(self) -> None:
        # Half 0: 8 lanes at 0x00, 8 lanes at 0x80 (both bank 0, 2 distinct dwords => 2-way)
        # Half 1: 4 lanes at 0x00, 4 at 0x80, 4 at 0x100, 4 at 0x180 (bank 0, 4 distinct dwords => 4-way)
        half0 = [0x00 if i < 8 else 0x80 for i in range(16)]
        half1 = [0x00, 0x80, 0x100, 0x180] * 4
        info = analyze_lds_access(Pipe.LDS_RD, tuple(half0 + half1))
        self.assertEqual((info.half0_max_way, info.half1_max_way), (2, 4))
        self.assertEqual(info.conflict_extra, (2 - 1) + (4 - 1))
        self.assertEqual(info.port_cycles, 2 + 4)

    def test_20_lds_exec_mask_eliminates_masked_half_conflicts(self) -> None:
        # Half 0 is conflict-free; Half 1 has 16-way bank conflict, but is masked off by EXEC
        half0 = [i * 4 for i in range(16)]
        half1 = [i * 128 for i in range(16)]
        addrs = tuple(half0 + half1)
        info_full = analyze_lds_access(Pipe.LDS_RD, addrs, exec_mask=0xFFFFFFFF)
        info_masked = analyze_lds_access(Pipe.LDS_RD, addrs, exec_mask=0x0000FFFF)
        self.assertEqual(info_full.conflict_extra, 15)
        self.assertEqual(info_masked.conflict_extra, 0)
        self.assertEqual(info_masked.port_cycles, 2)

    def test_21_lds_wr_1_to_5_dwords_times_conflict_factor(self) -> None:
        # 4-way conflict in each half-wave (stride 32 bytes = 8 dwords -> 4 distinct dwords/bank in 16 lanes)
        addrs_4way = tuple((lane % 4) * 128 + (lane // 4) * 4 for lane in range(WAVE_LANES))
        for pipe, dwords in LDS_WR_PIPES.items():
            info_clean = analyze_lds_access(pipe, tuple(i * 4 for i in range(WAVE_LANES)))
            self.assertEqual(info_clean.conflict_factor, 1)
            self.assertEqual(info_clean.port_cycles, dwords)

            info_4way = analyze_lds_access(pipe, addrs_4way)
            self.assertEqual(info_4way.conflict_factor, 4)
            self.assertEqual(info_4way.port_cycles, dwords * 4)

    def test_22_lgkmcnt_scoreboard_wait_0_and_wait_1(self) -> None:
        unit_addrs = tuple(i * 4 for i in range(WAVE_LANES))
        prog = [
            Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=unit_addrs),  # issue 0, done 20
            Instruction(pc=0x04, pipe=Pipe.LDS_RD, dst=("v2",), lds_addrs=unit_addrs),  # issue 2, done 22
            Instruction(pc=0x08, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=1),  # waits until 1st completes (t=20)
            Instruction(pc=0x0C, pipe=Pipe.VALU32, dst=("v3",), src=("v1",)),  # v1 ready at t=20 -> issues 21
            Instruction(pc=0x10, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),  # waits until 2nd completes (t=22)
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].to_tuple(), (0x00, 0, 0, 20))
        self.assertEqual(ev[1].to_tuple(), (0x04, 2, 0, 20))
        self.assertEqual(ev[2].issue_cycle, 20)
        self.assertEqual(ev[2].lgkm_inflight, 1)
        self.assertEqual(ev[3].issue_cycle, 21)
        self.assertEqual(ev[3].stall_cycles, 0)
        self.assertEqual(ev[4].issue_cycle, 22)
        self.assertEqual(ev[4].lgkm_inflight, 0)

    def test_23_smem_rd_increments_lgkmcnt_scoreboard(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SMEM_RD, dst=("s4",), src=("s0",)),  # issue 0, done 20
            Instruction(pc=0x04, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
            Instruction(pc=0x08, pipe=Pipe.SALU, dst=("s5",), src=("s4",)),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].lgkm_inflight, 1)
        self.assertEqual(ev[1].issue_cycle, 20)
        self.assertEqual(ev[1].stall_cycles, 19)
        self.assertEqual(ev[2].issue_cycle, 21)

    def test_24_cross_simd_other_lds_port_contention_stalls_issue_and_stretches_phases(self) -> None:
        unit_addrs = tuple(i * 4 for i in range(WAVE_LANES))
        # Background OTHER_LDS occupies cycles [0, 3) (stalling issue to t=3) and cycle [4, 6) (stretching 2nd phase)
        bg = [
            ContentionEvent(port=ContentionPort.OTHER_LDS, start_cycle=0, duration_cycles=3),
            ContentionEvent(port=ContentionPort.OTHER_LDS, start_cycle=4, duration_cycles=2),
        ]
        prog = [
            Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=unit_addrs),
            Instruction(pc=0x04, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
        ]
        ev = simulate_wave(prog, contention=bg)
        # Starts at t=3 (cycles 0,1,2 busy => stall=3); uses cycle 3, then cycles 4,5 stolen (+2), uses cycle 6
        self.assertEqual(ev[0].issue_cycle, 3)
        self.assertEqual(ev[0].stall_cycles, 3)
        self.assertEqual(ev[0].port_contention_cycles, 2)
        self.assertEqual(ev[0].duration_cycles, 22)
        self.assertEqual(ev[1].issue_cycle, 25)

    def test_25_cross_simd_other_vmem_port_contention(self) -> None:
        bg = [ContentionEvent(port=ContentionPort.OTHER_VMEM, start_cycle=0, duration_cycles=5)]
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VMEM_RD, dst=("v1",), src=("s0",)),
        ]
        ev = simulate_wave(prog, contention=bg)
        self.assertEqual(ev[0].issue_cycle, 5)
        self.assertEqual(ev[0].stall_cycles, 5)
        self.assertEqual(ev[0].duration_cycles, 120)

    def test_26_multi_wave_workgroup_barrier_synchronization(self) -> None:
        # Wave 0 arrives at barrier at t=1; Wave 1 runs VALUT_4 (dur=16) + dep VALU32 before barrier -> arrives t=17
        w0 = [
            Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s1",)),
            Instruction(pc=0x04, pipe=Pipe.BARRIER, barrier_id=1),
            Instruction(pc=0x08, pipe=Pipe.SALU, dst=("s2",)),
        ]
        w1 = [
            Instruction(pc=0x00, pipe=Pipe.VALUT_4, dst=("v1",)),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v2",), src=("v1",)),
            Instruction(pc=0x08, pipe=Pipe.BARRIER, barrier_id=1),
            Instruction(pc=0x0C, pipe=Pipe.SALU, dst=("s2",)),
        ]
        sim = Wave32Simulator()
        res = sim.run_workgroup([(0, 0, w0), (1, 1, w1)])
        # Wave 1 VALU32 issues at t=16, so barrier arrival is t=17; BOTH waves release barrier at t=17
        self.assertEqual(res[0][1].issue_cycle, 17)
        self.assertEqual(res[0][1].stall_cycles, 16)
        self.assertEqual(res[1][2].issue_cycle, 17)
        self.assertEqual(res[1][2].stall_cycles, 0)
        self.assertEqual(res[0][2].issue_cycle, 18)
        self.assertEqual(res[1][3].issue_cycle, 18)

    def test_27_multi_wave_shared_lds_port_arbitration_across_simds(self) -> None:
        unit_addrs = tuple(i * 4 for i in range(WAVE_LANES))
        w0 = [Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=unit_addrs)]
        w1 = [Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=unit_addrs)]
        sim = Wave32Simulator()
        res = sim.run_workgroup([(0, 0, w0), (1, 1, w1)])
        # Wave 0 reserves LDS port cycles [0, 2); Wave 1 on SIMD 1 stalls until t=2!
        self.assertEqual(res[0][0].issue_cycle, 0)
        self.assertEqual(res[1][0].issue_cycle, 2)
        self.assertEqual(res[1][0].stall_cycles, 2)

    def test_28_same_simd_multi_wave_issue_slot_sharing(self) -> None:
        w0 = [Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s1",))]
        w1 = [Instruction(pc=0x00, pipe=Pipe.SALU, dst=("s1",))]
        sim = Wave32Simulator()
        res = sim.run_workgroup([(0, 2, w0), (1, 2, w1)])
        # Both waves share SIMD 2 -> cannot issue in the exact same cycle
        self.assertEqual(res[0][0].issue_cycle, 0)
        self.assertEqual(res[1][0].issue_cycle, 1)
        self.assertEqual(res[1][0].stall_cycles, 1)

    def test_29_dynamic_salu_wr_exec_updates_subsequent_lds_mask(self) -> None:
        # Upper half has 16-way conflict; SALU_WR_EXEC masks off upper half before LDS_RD
        addrs = tuple([i * 4 for i in range(16)] + [i * 128 for i in range(16)])
        prog = [
            Instruction(pc=0x00, pipe=Pipe.SALU_WR_EXEC, exec_mask=0x0000FFFF),
            Instruction(pc=0x04, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=addrs),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[1].issue_cycle, 2)  # Waits for SALU_WR_EXEC (2 cyc)
        self.assertEqual(ev[1].bank_conflicts, 0)
        self.assertEqual(ev[1].lds_port_cycles, 2)

    def test_30_fit_profile_zero_mae_on_perturbed_synthetic_sqtt_traces(self) -> None:
        # Ground-truth custom hardware-like latency profile significantly different from defaults
        gt_profile = LatencyProfile(
            salu_issue=1,
            salu_latency=1,
            salu_wr_exec_issue=3,
            salu_wr_exec_latency=3,
            jump_taken_cycles=6,
            jump_untaken_cycles=2,
            valu32_issue=1,
            valu32_latency=5,
            valu32_forwarding=False,
            valut_4_issue=5,
            valut_4_latency=18,
            valub_2_issue=3,
            valub_2_latency=3,
            valub_4_issue=5,
            valub_4_latency=5,
            valub_16_issue=14,
            valub_16_latency=14,
            lds_rd_base=3,
            lds_rd_latency=24,
            lds_wr_latency_overhead=10,
            smem_rd_issue=1,
            smem_rd_latency=26,
            enable_dual_issue=True,
        )
        unit_addrs = tuple(i * 4 for i in range(WAVE_LANES))
        conflict_2way = tuple((i % 2) * 128 + (i // 2) * 4 for i in range(WAVE_LANES))

        prog_alu = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s2",), src=("s0",)),  # dual-issue!
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v2",), src=("v1",)),  # RAW valu32_latency
            Instruction(pc=0x0C, pipe=Pipe.VALUT_4, dst=("v3",), src=("v2",)),
            Instruction(pc=0x10, pipe=Pipe.VALU32, dst=("v4",), src=("v5",)),  # VALUT issue occupancy
            Instruction(pc=0x14, pipe=Pipe.VALU32, dst=("v6",), src=("v3",)),  # VALUT latency
            Instruction(pc=0x18, pipe=Pipe.VALUB_2, dst=("v7",), src=("v6",)),
            Instruction(pc=0x1C, pipe=Pipe.VALUB_4, dst=("v8",), src=("v7",)),
            Instruction(pc=0x20, pipe=Pipe.VALUB_16, dst=("v9",), src=("v8",)),
            Instruction(pc=0x24, pipe=Pipe.SALU_WR_EXEC, exec_mask=0xFFFFFFFF),
            Instruction(pc=0x28, pipe=Pipe.JUMP, taken=False),
            Instruction(pc=0x2C, pipe=Pipe.JUMP, taken=True),
        ]
        prog_mem = [
            Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=unit_addrs),
            Instruction(pc=0x04, pipe=Pipe.LDS_RD, dst=("v2",), lds_addrs=conflict_2way),
            Instruction(pc=0x08, pipe=Pipe.LDS_WR_3, src=("v0",), lds_addrs=unit_addrs),
            Instruction(pc=0x0C, pipe=Pipe.SMEM_RD, dst=("s4",), src=("s0",)),
            Instruction(pc=0x10, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
            Instruction(pc=0x14, pipe=Pipe.VALU32, dst=("v3",), src=("v1", "v2")),
        ]

        obs_alu = [e.to_sqtt() for e in simulate_wave(prog_alu, profile=gt_profile)]
        obs_mem = [e.to_sqtt() for e in simulate_wave(prog_mem, profile=gt_profile)]

        samples = [
            TraceSample(program=prog_alu, observed=obs_alu, sqtt_format=True),
            TraceSample(program=prog_mem, observed=obs_mem, sqtt_format=True),
        ]
        # Start from default profile (high initial error) and recover 0.0 MAE
        init_mae = _evaluate_profile_mae(LatencyProfile(), samples)
        self.assertGreater(init_mae, 1.0)

        fitted, mae = fit_profile(samples, initial_profile=LatencyProfile())
        self.assertEqual(mae, 0.0)
        self.assertEqual(fitted.valu32_latency, 5)
        self.assertEqual(fitted.valut_4_issue, 5)
        self.assertEqual(fitted.valut_4_latency, 18)
        self.assertEqual(fitted.lds_rd_base, 3)
        self.assertEqual(fitted.lds_rd_latency, 24)
        self.assertEqual(fitted.lds_wr_latency_overhead, 10)
        self.assertEqual(fitted.smem_rd_latency, 26)
        self.assertEqual(fitted.jump_taken_cycles, 6)

    def test_31_fit_profile_detects_disabled_dual_issue_and_forwarding(self) -> None:
        gt_profile = LatencyProfile(
            enable_dual_issue=False,
            valu32_forwarding=True,
            valu32_forward_latency=2,
            valu32_latency=4,
        )
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=0x04, pipe=Pipe.SALU, dst=("s2",), src=("s0",)),
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v2",), src=("v1",)),
            Instruction(pc=0x0C, pipe=Pipe.VALU32, dst=("v3",), src=("v2",)),
        ]
        obs = [e.to_sqtt() for e in simulate_wave(prog, profile=gt_profile)]
        fitted, mae = fit_profile([TraceSample(program=prog, observed=obs, sqtt_format=True)])
        self.assertEqual(mae, 0.0)
        self.assertFalse(fitted.enable_dual_issue)
        self.assertTrue(fitted.valu32_forwarding)
        self.assertEqual(fitted.valu32_forward_latency, 2)

    def test_32_deterministic_reproducibility_across_runs(self) -> None:
        addrs = tuple(((i * 7) % 32) * 4 + (i // 8) * 128 for i in range(WAVE_LANES))
        prog = [
            Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v1",), lds_addrs=addrs),
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v2",), src=("v0",)),
            Instruction(pc=0x08, pipe=Pipe.SALU, dst=("s2",), src=("s0",)),
            Instruction(pc=0x0C, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
            Instruction(pc=0x10, pipe=Pipe.VALUT_4, dst=("v3",), src=("v1", "v2")),
            Instruction(pc=0x14, pipe=Pipe.LDS_WR_4, src=("v3",), lds_addrs=addrs),
        ]
        bg = [ContentionEvent(port=ContentionPort.OTHER_LDS, start_cycle=1, duration_cycles=2)]
        run1 = [e.to_tuple() for e in simulate_wave(prog, contention=bg)]
        run2 = [e.to_tuple() for e in simulate_wave(prog, contention=bg)]
        self.assertEqual(run1, run2)

    def test_33_waw_protection_long_valut_followed_by_short_valu32(self) -> None:
        prog = [
            Instruction(pc=0x00, pipe=Pipe.VALUT_4, dst=("v1",), src=("v0",)),  # done at t=16
            Instruction(pc=0x04, pipe=Pipe.VALU32, dst=("v1",), src=("v2",)),   # dur=4 => must wait until t=12
        ]
        ev = simulate_wave(prog)
        self.assertEqual(ev[0].completion_cycle, 16)
        self.assertEqual(ev[1].issue_cycle, 12)
        self.assertEqual(ev[1].completion_cycle, 16)

    def test_34_full_sram_gemm_microkernel_pipeline_trace(self) -> None:
        # Realistic SRAM-tiled Wave32 GEMM loop iteration:
        # 2x LDS_RD (A & B tiles) + overlapped dual-issue VALU/SALU address math +
        # wait_lgkmcnt(1) + FMA chain + wait_lgkmcnt(0) + LDS_WR_2 + barrier
        a_addrs = tuple(i * 4 for i in range(WAVE_LANES))
        b_addrs = tuple((i % 4) * 4 for i in range(WAVE_LANES))  # broadcast groups, 0 conflict
        prog = [
            Instruction(pc=0x00, pipe=Pipe.LDS_RD, dst=("v_a",), lds_addrs=a_addrs),
            Instruction(pc=0x04, pipe=Pipe.LDS_RD, dst=("v_b",), lds_addrs=b_addrs),
            Instruction(pc=0x08, pipe=Pipe.VALU32, dst=("v_acc0",), src=("v_acc0",)),
            Instruction(pc=0x0C, pipe=Pipe.SALU, dst=("s_k",), src=("s_k",)),  # dual-issue with 0x08
            Instruction(pc=0x10, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=1),
            Instruction(pc=0x14, pipe=Pipe.VALU32, dst=("v_acc1",), src=("v_a",)),
            Instruction(pc=0x18, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
            Instruction(pc=0x1C, pipe=Pipe.VALU32, dst=("v_acc2",), src=("v_b", "v_acc1")),
            Instruction(pc=0x20, pipe=Pipe.LDS_WR_2, src=("v_acc2",), lds_addrs=a_addrs),
            Instruction(pc=0x24, pipe=Pipe.WAIT_LGKMCNT, lgkm_target=0),
        ]
        ev = simulate_wave(prog)
        self.assertEqual(len(ev), 10)
        self.assertEqual(ev[0].bank_conflicts, 0)
        self.assertEqual(ev[1].bank_conflicts, 0)
        self.assertTrue(ev[2].dual_issued and ev[3].dual_issued)
        self.assertEqual(ev[2].issue_cycle, ev[3].issue_cycle)
        self.assertEqual(ev[4].issue_cycle, 20)  # v_a done at t=20
        self.assertEqual(ev[6].issue_cycle, 22)  # v_b done at t=22
        self.assertEqual(ev[7].issue_cycle, 25)  # v_acc1 issued at 21 + lat 4 = 25
        self.assertEqual(ev[8].issue_cycle, 29)  # v_acc2 done at 25 + 4 = 29
        self.assertEqual(ev[9].issue_cycle, 39)  # LDS_WR_2 dur=2+8=10 => done 39

    def test_35_input_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            analyze_lds_access(Pipe.LDS_RD, (0, 4, 8))  # wrong lane count
        with self.assertRaises(ValueError):
            ContentionEvent(port=ContentionPort.OTHER_LDS, start_cycle=-1, duration_cycles=2)
        with self.assertRaises(ValueError):
            Wave32Simulator().run_workgroup([(0, 4, [])])  # simd_id out of 0..3


# ---------------------------------------------------------------------------
# CLI Self-Test & Demo Runner
# ---------------------------------------------------------------------------


class TestGFX11Rev2AndSQTT(unittest.TestCase):
    """DISPATCH-09 REV=2 verification: raw GFX1100 binary decode, SQTT nibble codec,
    defect fixes (VALU-only bypass, multi-dword DS load/store, SMEM zero-LDS, dynamic
    multi-wave CU arbitration), and binary GFX11+SQTT fit_profile calibration.
    """

    def test_36_decode_real_clang21_gfx1100_all_encodings(self) -> None:
        # Ground-truth little-endian words assembled by LLVM/Clang 21 (-mcpu=gfx1100)
        words = [
            0x80020604,              # s_add_u32 s2, s4, s6 (SOP2)
            0x8B7E0604,              # s_and_b32 exec_lo, s4, s6 (SOP2 -> SALU_WR_EXEC)
            0xBEFE00FF, 0x0000FFFF,  # s_mov_b32 exec_lo, 0x0000ffff (SOP1 + lit -> SALU_WR_EXEC)
            0xB0021234,              # s_movk_i32 s2, 0x1234 (SOPK)
            0xBF060402,              # s_cmp_eq_u32 s2, s4 (SOPC)
            0xBF890817,              # s_waitcnt vmcnt(2) lgkmcnt(1) (SOPP)
            0xBFBD0000,              # s_barrier (SOPP)
            0xBFA00004,              # s_branch +4 (SOPP)
            0x06000501,              # v_add_f32 v0, v1, v2 (VOP2)
            0x5A000501, 0x3F800000,  # v_fmaak_f32 v0, v1, v2, 0x3f800000 (VOP2 + lit)
            0x58000501, 0x3F800000,  # v_fmamk_f32 v0, v1, 0x3f800000, v2 (VOP2 + lit)
            0x7E0002FF, 0x80000000,  # v_mov_b32 v0, 0x80000000 (VOP1 + lit)
            0x7E065504,              # v_rcp_f32 v3, v4 (VOP1 -> VALUT_4)
            0x7E066704,              # v_sqrt_f32 v3, v4 (VOP1 -> VALUT_4)
            0x7E066B04,              # v_sin_f32 v3, v4 (VOP1 -> VALUT_4)
            0xD6130005, 0x04220F06,  # v_fma_f32 v5, v6, v7, v8 (VOP3 -> VALU32)
            0xD73C0004, 0x00020C82,  # v_lshlrev_b64 v[4:5], 2, v[6:7] (VOP3 -> VALUB_2)
            0xD72C0005, 0x00020F06,  # v_mul_lo_u32 v5, v6, v7 (VOP3 -> VALUB_4)
            0xD7270004, 0x00021106,  # v_add_f64 v[4:5], v[6:7], v[8:9] (VOP3 -> VALUB_16)
            0xD6140004, 0x042A1106,  # v_fma_f64 v[4:5], v[6:7], v[8:9], v[10:11] (VOP3 -> VALUB_16)
            0xC9060501, 0x00020B04,  # v_dual_add_f32 v0,v1,v2 :: v_dual_mul_f32 v3,v4,v5 (VOPD)
            0xC8480501, 0x00020B04, 0x3F800000,  # v_dual_fmaak_f32 :: v_dual_add_f32 (VOPD + lit)
            0xD8D80010, 0x0900000A,  # ds_load_b32 v9, v10 offset:16 (DS)
            0xDBFC0040, 0x0800000A,  # ds_load_b128 v[8:11], v10 offset:64 (DS)
            0xDB7C0000, 0x00000C0A,  # ds_store_b128 v10, v[12:15] (DS)
            0xF4080100, 0xF8000040,  # s_load_b128 s[4:7], s[0:1], 0x40 (SMEM)
            0xDC520000, 0x027C0004,  # global_load_b32 v2, v[4:5], off (GLOBAL)
            0xDC6E0000, 0x007C0204,  # global_store_b64 v[4:5], v[2:3], off (GLOBAL)
            0xBFB00000,              # s_endpgm (SOPP)
        ]
        blob = struct.pack(f"<{len(words)}I", *words)
        dec = decode_gfx11(blob)
        self.assertEqual(len(dec), 29)
        self.assertEqual(dec[-1].pc + dec[-1].size, len(blob))

        # Check key decoded semantics
        self.assertEqual((dec[0].encoding, dec[0].mnemonic, dec[0].pipe), ("SOP2", "s_add_u32", Pipe.SALU))
        self.assertEqual((dec[1].mnemonic, dec[1].pipe, dec[1].writes_exec), ("s_and_b32", Pipe.SALU_WR_EXEC, True))
        self.assertEqual((dec[2].mnemonic, dec[2].pipe, dec[2].exec_mask, dec[2].size), ("s_mov_b32", Pipe.SALU_WR_EXEC, 0xFFFF, 8))
        self.assertEqual((dec[3].encoding, dec[3].mnemonic, dec[3].simm16), ("SOPK", "s_movk_i32", 0x1234))
        self.assertEqual((dec[4].encoding, dec[4].mnemonic, dec[4].dst), ("SOPC", "s_cmp_eq_u32", ("scc",)))
        self.assertEqual((dec[5].mnemonic, dec[5].vmcnt, dec[5].lgkmcnt, dec[5].expcnt), ("s_waitcnt", 2, 1, 7))
        self.assertEqual((dec[6].mnemonic, dec[6].pipe), ("s_barrier", Pipe.BARRIER))
        self.assertEqual((dec[7].mnemonic, dec[7].pipe, dec[7].branch_target), ("s_branch", Pipe.JUMP, dec[7].pc + 4 + 16))
        self.assertEqual((dec[8].mnemonic, dec[8].dst, dec[8].src), ("v_add_f32", ("v0",), ("v1", "v2")))
        self.assertEqual((dec[9].mnemonic, dec[9].size, dec[9].literal), ("v_fmaak_f32", 8, 0x3F800000))
        self.assertEqual((dec[10].mnemonic, dec[10].size, dec[10].literal), ("v_fmamk_f32", 8, 0x3F800000))
        self.assertEqual((dec[11].mnemonic, dec[11].literal), ("v_mov_b32", 0x80000000))
        self.assertEqual([d.pipe for d in dec[12:15]], [Pipe.VALUT_4, Pipe.VALUT_4, Pipe.VALUT_4])
        self.assertEqual((dec[15].mnemonic, dec[15].pipe), ("v_fma_f32", Pipe.VALU32))
        self.assertEqual((dec[16].mnemonic, dec[16].pipe, dec[16].dst), ("v_lshlrev_b64", Pipe.VALUB_2, ("v4", "v5")))
        self.assertEqual((dec[17].mnemonic, dec[17].pipe), ("v_mul_lo_u32", Pipe.VALUB_4))
        self.assertEqual((dec[18].mnemonic, dec[18].pipe), ("v_add_f64", Pipe.VALUB_16))
        self.assertEqual((dec[19].mnemonic, dec[19].pipe), ("v_fma_f64", Pipe.VALUB_16))
        self.assertTrue(dec[20].is_vopd and dec[20].vopd_bank_ok)
        self.assertEqual((dec[20].vopd_vdstx, dec[20].vopd_vdsty), (0, 3))
        self.assertEqual((dec[21].size, dec[21].literal), (12, 0x3F800000))
        self.assertEqual((dec[22].mnemonic, dec[22].lds_dwords, dec[22].lds_offset, dec[22].lds_addr_reg), ("ds_load_b32", 1, 16, "v10"))
        self.assertEqual((dec[23].mnemonic, dec[23].lds_dwords, dec[23].lds_offset), ("ds_load_b128", 4, 64))
        self.assertEqual((dec[24].mnemonic, dec[24].pipe, dec[24].lds_dwords), ("ds_store_b128", Pipe.LDS_WR_4, 4))
        self.assertEqual((dec[25].mnemonic, dec[25].pipe, dec[25].dst), ("s_load_b128", Pipe.SMEM_RD, ("s4", "s5", "s6", "s7")))
        self.assertEqual((dec[26].mnemonic, dec[26].pipe), ("global_load_b32", Pipe.VMEM_RD))
        self.assertEqual((dec[27].mnemonic, dec[27].pipe), ("global_store_b64", Pipe.VMEM_WR))
        self.assertEqual((dec[28].mnemonic, dec[28].pipe), ("s_endpgm", Pipe.ENDPGM))

    def test_37_encoders_match_clang21_byte_for_byte(self) -> None:
        self.assertEqual(encode_s_waitcnt(vmcnt=2, lgkmcnt=1, expcnt=7), struct.pack("<I", 0xBF890817))
        self.assertEqual(encode_s_waitcnt(lgkmcnt=0), struct.pack("<I", 0xBF89FC07))
        self.assertEqual(encode_s_waitcnt(vmcnt=0), struct.pack("<I", 0xBF8903F7))
        self.assertEqual(encode_s_barrier(), struct.pack("<I", 0xBFBD0000))
        self.assertEqual(encode_s_endpgm(), struct.pack("<I", 0xBFB00000))
        self.assertEqual(encode_s_branch(-2, cond_op=0x26), struct.pack("<I", 0xBFA6FFFE))
        self.assertEqual(encode_sop2(0x16, EXEC_LO_ENC, 4, 6), struct.pack("<I", 0x8B7E0604))
        self.assertEqual(encode_sop1(0x00, EXEC_LO_ENC, LITERAL_ENC, 0xFFFF), struct.pack("<II", 0xBEFE00FF, 0xFFFF))
        self.assertEqual(encode_vop2(0x03, 0, 257, 2), struct.pack("<I", 0x06000501))
        self.assertEqual(encode_vop2(0x2D, 0, 257, 2, 0x3F800000), struct.pack("<II", 0x5A000501, 0x3F800000))
        self.assertEqual(encode_vop1(0x2A, 3, 260), struct.pack("<I", 0x7E065504))
        self.assertEqual(encode_vop3(0x213, 5, 262, 263, 264), struct.pack("<II", 0xD6130005, 0x04220F06))
        self.assertEqual(encode_vopd(0x04, 0x03, 0, 257, 2, 3, 260, 5), struct.pack("<II", 0xC9060501, 0x00020B04))
        self.assertEqual(encode_ds_offset16(0x36, 10, 0, 9, 16), struct.pack("<II", 0xD8D80010, 0x0900000A))
        self.assertEqual(encode_ds_offset16(0xDF, 10, 12, 0, 0), struct.pack("<II", 0xDB7C0000, 0x00000C0A))
        self.assertEqual(encode_smem(0x00, 4, 0, 0x10), struct.pack("<II", 0xF4000100, 0xF8000010))
        self.assertEqual(encode_global(0x14, 4, 0, 2), struct.pack("<II", 0xDC520000, 0x027C0004))

    def test_38_s_waitcnt_vmcnt_vs_lgkmcnt_from_raw_bytes(self) -> None:
        code = (
            encode_global(0x14, 4, 0, 2)           # global_load_b32 v2, v[4:5], off (VMEM_RD: issue 0, done 120)
            + encode_ds_offset16(0x36, 10, 0, 9)   # ds_load_b32 v9, v10 (LDS_RD: issue 4, done 24)
            + encode_s_waitcnt(lgkmcnt=0)          # waits ONLY on LDS_RD -> t=24
            + encode_s_waitcnt(vmcnt=0)            # waits on GLOBAL_LOAD -> t=120
            + encode_s_endpgm()
        )
        ev = simulate_wave(code)
        self.assertEqual(ev[0].pipe, Pipe.VMEM_RD)
        self.assertEqual(ev[1].pipe, Pipe.LDS_RD)
        self.assertEqual(ev[2].issue_cycle, 24)
        self.assertEqual(ev[3].issue_cycle, 120)
        self.assertEqual(ev[4].issue_cycle, 121)

    def test_39_branch_target_signed_pc_relative(self) -> None:
        code = (
            encode_s_branch(4, cond_op=0x20)      # pc=0, target=0+4+16=20
            + encode_s_branch(-2, cond_op=0x26)   # pc=4, target=4+4-8=0
        )
        dec = decode_gfx11(code)
        self.assertEqual(dec[0].branch_target, 20)
        self.assertEqual(dec[1].branch_target, 0)
        self.assertEqual(dec[1].src, (EXEC_REG,))

    def test_40_salu_wr_exec_binary_masks_subsequent_lds_conflicts(self) -> None:
        # Half 0 conflict-free, Half 1 16-way conflict; s_mov_b32 exec_lo, 0x0000ffff disables Half 1
        addrs = [i * 4 for i in range(16)] + [i * 128 for i in range(16)]
        code = (
            encode_sop1(0x00, EXEC_LO_ENC, LITERAL_ENC, 0x0000FFFF)
            + encode_ds_offset16(0x36, 10, 0, 9)
        )
        ev = simulate_wave(code, vgpr_state={"v10": addrs})
        self.assertEqual(ev[0].pipe, Pipe.SALU_WR_EXEC)
        self.assertEqual(ev[0].duration_cycles, 2)
        self.assertEqual(ev[1].issue_cycle, 2)
        self.assertEqual(ev[1].bank_conflicts, 0)
        self.assertEqual(ev[1].lds_port_cycles, 2)

    def test_41_vopd_dual_issue_and_bank_parity_enforcement(self) -> None:
        with self.assertRaises(ValueError):
            encode_vopd(0x04, 0x03, 0, 257, 2, 2, 260, 5)  # both even -> illegal!
        code = (
            encode_vopd(0x04, 0x03, 0, 257, 2, 3, 260, 5)  # v0 (even) & v3 (odd)
            + encode_vop2(0x03, 6, 256, 3)                  # reads v0 and v3 -> stalls 3 cycles (raw lat 4)
        )
        ev = simulate_wave(code)
        self.assertEqual(ev[0].alu_src, AluSrc.VOPD)
        self.assertTrue(ev[0].dual_issued)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 4)
        self.assertEqual(ev[1].stall_cycles, 3)

    def test_42_valu32_bypass_restricted_to_valu_consumers_only(self) -> None:
        # REV=2 CRITICAL FIX: even when profile.valu32_forwarding=True,
        # VALU32 -> VALU32 gets 1-cyc bypass, while VALU32 -> DS/VMEM/SALU/JUMP MUST wait full 4 cyc!
        prof = LatencyProfile(valu32_forwarding=True, valu32_forward_latency=1, valu32_latency=4)
        # 1) VALU32 -> VALU32: bypass allowed (issue 0 -> 1, stall 0)
        ev_vv = simulate_wave([
            Instruction(pc=0, pipe=Pipe.VALU32, dst=("v1",), src=("v0",)),
            Instruction(pc=4, pipe=Pipe.VALU32, dst=("v2",), src=("v1",), forwarded=True),
        ], profile=prof)
        self.assertEqual(ev_vv[1].issue_cycle, 1)

        # 2) VALU32 -> DS (LDS address operand): bypass FORBIDDEN (must wait until t=4!)
        ev_ds = simulate_wave([
            Instruction(pc=0, pipe=Pipe.VALU32, dst=("v10",), src=("v0",)),
            Instruction(pc=4, pipe=Pipe.LDS_RD, dst=("v1",), src=("v10",), forwarded=True),
        ], profile=prof)
        self.assertEqual(ev_ds[1].issue_cycle, 4)
        self.assertEqual(ev_ds[1].stall_cycles, 3)

        # 3) VALU32 -> VMEM: bypass FORBIDDEN (t=4)
        ev_vm = simulate_wave([
            Instruction(pc=0, pipe=Pipe.VALU32, dst=("v4",), src=("v0",)),
            Instruction(pc=4, pipe=Pipe.VMEM_RD, dst=("v2",), src=("v4",), forwarded=True),
        ], profile=prof)
        self.assertEqual(ev_vm[1].issue_cycle, 4)

        # 4) VALU32 -> SALU / JUMP (via vcc_lo): bypass FORBIDDEN (t=4)
        ev_jmp = simulate_wave([
            Instruction(pc=0, pipe=Pipe.VALU32, dst=("vcc_lo",), src=("v0",)),
            Instruction(pc=4, pipe=Pipe.JUMP, src=("vcc_lo",), taken=True, forwarded=True),
        ], profile=prof)
        self.assertEqual(ev_jmp[1].issue_cycle, 4)

    def test_43_multi_dword_ds_load_and_store_b64_b96_b128_and_2addr(self) -> None:
        # 4-way bank conflict per half-wave
        addrs_4way = tuple((lane % 4) * 128 + (lane // 4) * 4 for lane in range(WAVE_LANES))
        code = (
            encode_ds_offset16(0x76, 10, 0, 4, 0)    # ds_load_b64 (2 dwords)
            + encode_ds_offset16(0xFE, 10, 0, 6, 0)  # ds_load_b96 (3 dwords)
            + encode_ds_offset16(0xFF, 10, 0, 12, 0) # ds_load_b128 (4 dwords)
            + encode_ds_offset16(0x4D, 10, 4, 0, 0)  # ds_store_b64 (2 dwords)
            + encode_ds_offset16(0xDE, 10, 6, 0, 0)  # ds_store_b96 (3 dwords)
            + encode_ds_offset16(0xDF, 10, 12, 0, 0) # ds_store_b128 (4 dwords)
        )
        ev = simulate_wave(code, vgpr_state={"v10": addrs_4way})
        # Loads: each dword has extra=(4-1)+(4-1)=6; port = 2*d + 6*d = 8*d
        self.assertEqual([e.lds_port_cycles for e in ev[:3]], [16, 24, 32])
        self.assertEqual([e.bank_conflicts for e in ev[:3]], [12, 18, 24])
        # Stores: each dword has max_way=4 -> port = 4*d, extra = 3*d
        self.assertEqual([e.lds_port_cycles for e in ev[3:6]], [8, 12, 16])
        self.assertEqual([e.bank_conflicts for e in ev[3:6]], [6, 9, 12])

        # Distinct bytes inside SAME 4B dword -> ZERO conflict even for ds_load_b128 & ds_store_b128
        sub_byte_addrs = tuple(0x40 + (lane % 4) for lane in range(WAVE_LANES))
        ev_sub = simulate_wave(
            encode_ds_offset16(0xFF, 10, 0, 8) + encode_ds_offset16(0xDF, 10, 8, 0),
            vgpr_state={"v10": sub_byte_addrs},
        )
        self.assertEqual(ev_sub[0].bank_conflicts, 0)
        self.assertEqual(ev_sub[1].bank_conflicts, 0)

        # 2addr offset0 / offset1 support
        dec_2a = decode_gfx11(encode_ds(0x37, 10, 0, 0, 8, offset0=1, offset1=33))
        self.assertTrue(dec_2a[0].lds_is_2addr)
        self.assertEqual((dec_2a[0].lds_offset0, dec_2a[0].lds_offset1), (1, 33))

    def test_44_smem_rd_zero_lds_crossbar_and_zero_lds_port_contention(self) -> None:
        # Wave 0 executes s_load_b128 (SMEM_RD); Wave 1 executes ds_load_b128 at t=0.
        # SMEM_RD must NEVER touch LDS crossbar or block Wave 1's LDS port,
        # yet MUST increment Wave 0's lgkmcnt scoreboard!
        w0 = WaveSpec(
            program=encode_smem(0x02, 4, 0, 0x40) + encode_s_waitcnt(lgkmcnt=0),
            wave_id=0, simd_id=0,
        )
        w1 = WaveSpec(
            program=encode_ds_offset16(0xFF, 10, 0, 8) + encode_s_waitcnt(lgkmcnt=0),
            wave_id=1, simd_id=1,
            vgpr_state={"v10": [i * 4 for i in range(WAVE_LANES)]},
        )
        res = simulate_cu([w0, w1])
        self.assertEqual(res[0][0].bank_conflicts, 0)
        self.assertEqual(res[0][0].lds_port_cycles, 0)
        self.assertEqual(res[0][0].lgkm_inflight, 1)
        self.assertEqual(res[0][1].issue_cycle, 20)
        # Wave 1 LDS_RD issues immediately at cycle 0 with ZERO contention from Wave 0 SMEM_RD!
        self.assertEqual(res[1][0].issue_cycle, 0)
        self.assertEqual(res[1][0].stall_cycles, 0)
        self.assertEqual(res[1][0].lds_port_cycles, 8)

    def test_45_simulate_cu_dynamic_shared_lds_and_vmem_port_arbitration(self) -> None:
        # 4 concurrent binary waves across SIMDs 0..3 issuing ds_load_b64 (4 port cycles each)
        # followed by global_load_b32 (4 VMEM port cycles each) and s_barrier.
        unit = [i * 4 for i in range(WAVE_LANES)]
        code = (
            encode_ds_offset16(0x76, 10, 0, 4)   # ds_load_b64 (4 LDS port cycles)
            + encode_global(0x14, 8, 0, 2)       # global_load_b32 (4 VMEM port cycles)
            + encode_s_barrier()
            + encode_s_endpgm()
        )
        waves = [WaveSpec(program=code, wave_id=i, simd_id=i, vgpr_state={"v10": unit}) for i in range(4)]
        res = simulate_cu(waves)
        # Dynamic shared LDS port serializes the 4 waves: 0, 4, 8, 12!
        self.assertEqual([res[i][0].issue_cycle for i in range(4)], [0, 4, 8, 12])
        # Dynamic shared VMEM port serializes their global loads: 4, 8, 12, 16!
        self.assertEqual([res[i][1].issue_cycle for i in range(4)], [4, 8, 12, 16])
        # All 4 waves release from s_barrier together at max arrival (wave 3 arrives at 16+4=20)
        self.assertEqual([res[i][2].issue_cycle for i in range(4)], [20, 20, 20, 20])
    def test_46_real_gfx11_sqtt_v2_codec_roundtrip_binary_and_hex(self) -> None:
        code = (
            encode_vop2(0x03, 0, 257, 2)
            + encode_sop2(0x00, 2, 4, 6)
            + encode_vop1(0x2A, 3, 260)
            + encode_ds_offset16(0x36, 10, 0, 9, 16)
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_s_endpgm()
        )
        ev = simulate_wave(
            code, wave_id=3, simd_id=2,
            vgpr_state={"v10": [i * 128 for i in range(WAVE_LANES)]},
            hw_sqtt_mode=True,
        )
        raw_sqtt = encode_sqtt(ev, base_timestamp=10000, cu_id=5, simd_id=2)
        pkts = parse_sqtt(raw_sqtt)
        kinds = [p.kind for p in pkts]
        self.assertIn(SQTTPacketKind.LAYOUT_HEADER, kinds)
        self.assertIn(SQTTPacketKind.WAVESTART, kinds)
        self.assertIn(SQTTPacketKind.VALUINST, kinds)
        self.assertIn(SQTTPacketKind.INST, kinds)
        self.assertIn(SQTTPacketKind.IMMEDIATE, kinds)
        self.assertIn(SQTTPacketKind.WAVEEND, kinds)

        # Hex string & binary ingest align identically to exact GFX11 PCs & issue cycles
        hex_str = " ".join(f"{b:02x}" for b in raw_sqtt)
        tuples_bin = sqtt_to_tuples(raw_sqtt, program=code, cu=5)[3]
        tuples_hex = sqtt_to_tuples(hex_str, program=code, cu=5)[3]
        self.assertEqual(tuples_bin, tuples_hex)
        self.assertEqual([(t[0], t[3]) for t in tuples_bin], [(e.pc, e.issue_cycle) for e in ev])

    def test_47_parse_real_rdna3_hardware_sqtt_packets_bit_exact(self) -> None:
        # Hand-assembled genuine AMD RDNA3 (GFX11 Layout 3) hardware SQTT v2 nibble stream:
        # LAYOUT_HEADER(layout=3, simd=0) + TS_DELTA_OR_MARK(100) + WAVESTART(cu=0,simd=0,w=1)
        # + VALUINST(w=1, dt=1) + IMMEDIATE(w=1, dt=2) + INST(w=1, op=LDS_RD=0x29, dt=5)
        # + TS_DELTA_SHORT(dt=10) + ALUEXEC(VALU_SALU) + VMEMEXEC(LDS) + WAVEEND
        w = _SQTTNibbleWriter()
        w.emit(0b0010001 | (3 << 7) | (0 << 13), 16)                    # LAYOUT_HEADER
        w.emit(0b0000001 | (100 << 12), 12)                             # TS_DELTA_OR_MARK (+100)
        w.emit(0b01100 | (0 << 5) | (0 << 7) | (0 << 8) | (0 << 10) | (1 << 13), 8)  # WAVESTART
        w.emit(0b011 | (1 << 3) | (1 << 7), 3)                          # VALUINST (+1 -> 101)
        w.emit(0b1101 | (2 << 4) | (1 << 7), 3)                         # IMMEDIATE (+2 -> 103)
        w.emit(0b010 | (5 << 4) | (1 << 8) | (0x29 << 13), 5)           # INST LDS_RD (+5 -> 108)
        w.emit(0b1000 | ((10 - 4) << 4), 2)                             # TS_DELTA_SHORT (+10 -> 118)
        w.emit(0b1110 | (SQTTAluSrc.VALU_SALU.value << 6), 2)           # ALUEXEC VALU_SALU
        w.emit(0b1111 | (SQTTMemSrc.LDS.value << 6), 2)                 # VMEMEXEC LDS
        w.emit(0b10101 | (1 << 5) | (1 << 15), 5)                       # WAVEEND (+1 -> 119)

        pkts = parse_sqtt(w.to_bytes().hex())
        inst_pkts = [p for p in pkts if p.kind in _SQTT_INST_KINDS]
        self.assertEqual(len(inst_pkts), 3)
        self.assertEqual((inst_pkts[0].kind, inst_pkts[0].timestamp, inst_pkts[0].wave_id), (SQTTPacketKind.VALUINST, 101, 1))
        self.assertEqual((inst_pkts[1].kind, inst_pkts[1].timestamp, inst_pkts[1].wave_id), (SQTTPacketKind.IMMEDIATE, 103, 1))
        self.assertEqual((inst_pkts[2].kind, inst_pkts[2].timestamp, inst_pkts[2].op_name), (SQTTPacketKind.INST, 108, "LDS_RD"))
        alu_pkts = [p for p in pkts if p.kind is SQTTPacketKind.ALUEXEC]
        vmem_pkts = [p for p in pkts if p.kind is SQTTPacketKind.VMEMEXEC]
        self.assertEqual(alu_pkts[0].src_name, "VALU_SALU")
        self.assertEqual(alu_pkts[0].timestamp, 118)
        self.assertEqual(vmem_pkts[0].src_name, "LDS")

    def test_48_fit_profile_end_to_end_raw_gfx11_and_real_sqtt_v2_zero_mae(self) -> None:
        hidden = LatencyProfile(
            salu_issue=2, salu_latency=2,
            salu_wr_exec_issue=3, salu_wr_exec_latency=3,
            valu32_issue=1, valu32_latency=6, valu32_forwarding=True, valu32_forward_latency=2,
            valut_4_issue=5, valut_4_latency=19,
            valub_2_issue=3, valub_2_latency=3,
            valub_4_issue=5, valub_4_latency=5,
            valub_16_issue=18, valub_16_latency=18,
            lds_rd_base=3, lds_rd_latency=24, lds_wr_latency_overhead=11,
            smem_rd_issue=1, smem_rd_latency=26,
            vmem_rd_issue=5, vmem_rd_latency=135,
            vmem_wr_issue=4, vmem_wr_latency=48,
            enable_dual_issue=True,
        )
        unit = [i * 4 for i in range(WAVE_LANES)]
        k1 = (
            encode_vop2(0x03, 1, 256, 0)                # v_add_f32 v1, v0, v0 (VALU32)
            + encode_sop2(0x00, 2, 4, 6)                # s_add_u32 s2, s4, s6 (SALU dual-issue)
            + encode_vop2(0x08, 2, 257, 1)              # v_mul_f32 v2, v1, v1 (VALU32 RAW -> fwd=2)
            + encode_ds_offset16(0x0D, 10, 2, 0)        # ds_store_b32 v10, v2 (non-VALU consumer -> exposes valu32_latency=6!)
            + encode_vop1(0x2A, 3, 258)                 # v_rcp_f32 v3, v2 (VALUT_4)
            + encode_vop1(0x33, 4, 256)                 # v_sqrt_f32 v4, v0 (VALUT_4 back-to-back)
            + encode_vop3(0x33C, 6, 130, 260)           # v_lshlrev_b64 v[6:7], 2, v[4:5] (VALUB_2)
            + encode_vop3(0x32C, 8, 262, 263)           # v_mul_lo_u32 v8, v6, v7 (VALUB_4)
            + encode_vop3(0x327, 10, 264, 266)          # v_add_f64 v[10:11], v[8:9], v[10:11] (VALUB_16)
            + encode_sop1(0x00, EXEC_LO_ENC, LITERAL_ENC, 0x0000FFFF)
            + encode_vop2(0x03, 12, 266, 12)            # observes VALUB_16 + SALU_WR_EXEC!
            + encode_s_endpgm()
        )
        k2 = (
            encode_ds_offset16(0x36, 10, 0, 1)          # ds_load_b32
            + encode_ds_offset16(0x36, 10, 0, 2)        # ds_load_b32 back-to-back (exposes lds_rd_base!)
            + encode_ds_offset16(0x4D, 10, 4, 0)        # ds_store_b64
            + encode_smem(0x00, 4, 0, 0x10)             # s_load_b32
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_global(0x14, 12, 0, 14)            # global_load_b32
            + encode_global(0x1A, 12, 14, 0)            # global_store_b32
            + encode_s_waitcnt(vmcnt=0)
            + encode_s_endpgm()
        )
        # Encode genuine AMD GFX11 SQTT v2 hardware streams (issue timestamps only, no synthetic EXT_INST!)
        sqtt1_hex = encode_sqtt(simulate_wave(k1, profile=hidden, vgpr_state={"v10": unit}, hw_sqtt_mode=True)).hex()
        sqtt2_bytes = encode_sqtt(simulate_wave(k2, profile=hidden, vgpr_state={"v10": unit}, hw_sqtt_mode=True))

        fitted, mae = fit_profile([
            TraceSample(program=k1, observed=sqtt1_hex, vgpr_state={"v10": unit}, hw_sqtt_mode=True),
            TraceSample(program=k2.hex(), observed=sqtt2_bytes, vgpr_state={"v10": unit}, hw_sqtt_mode=True),
        ])
        self.assertEqual(mae, 0.0)
        self.assertEqual(fitted.valu32_latency, hidden.valu32_latency)
        self.assertEqual(fitted.valu32_forwarding, hidden.valu32_forwarding)
        self.assertEqual(fitted.valu32_forward_latency, hidden.valu32_forward_latency)
        self.assertEqual(fitted.valut_4_issue, hidden.valut_4_issue)
        self.assertEqual(fitted.valut_4_latency, hidden.valut_4_latency)
        self.assertEqual(fitted.lds_rd_base, hidden.lds_rd_base)
        self.assertEqual(fitted.smem_rd_latency, hidden.smem_rd_latency)
        self.assertEqual(fitted.vmem_rd_issue, hidden.vmem_rd_issue)
        self.assertEqual(fitted.vmem_rd_latency, hidden.vmem_rd_latency)

    def test_49_decode_error_handling(self) -> None:
        with self.assertRaises(ValueError):
            decode_gfx11(b"\x00\x01\x02")  # non-multiple of 4 bytes
        with self.assertRaises(ValueError):
            decode_gfx11(struct.pack("<I", 0xD6130005))  # truncated 64-bit VOP3
        with self.assertRaises(ValueError):
            parse_sqtt("abc")  # odd hex digits

    def test_50_cli_decode_and_sqtt_parse(self) -> None:
        blob_hex = (encode_vop2(0x03, 0, 257, 2) + encode_s_waitcnt(lgkmcnt=0) + encode_s_endpgm()).hex()
        dec = decode_gfx11(blob_hex)
        self.assertEqual([d.mnemonic for d in dec], ["v_add_f32", "s_waitcnt", "s_endpgm"])
        sqtt_hex = encode_sqtt(simulate_wave(blob_hex, hw_sqtt_mode=True)).hex()
        self.assertEqual(len(sqtt_to_tuples(sqtt_hex, program=blob_hex)[0]), 2)


class TestGFX11Rev3RealHardwareSQTTAndISA(unittest.TestCase):
    """DISPATCH-09 REV=3 verification:
      1) Complete GFX11 ISA coverage (VOP3P/WMMA, MUBUF, SOPK s_waitcnt_vscnt, VOPCX -> exec_lo).
      2) Native AMDGPU ELF64 parsing + genuine AMD GFX11 SQTT v2 hardware packet decoding.
      3) Real 32-lane Wave32 functional ISA execution (no manual vgpr_state/branch_outcomes/forwarded_pcs).
      4) Real AMD Radeon RX 7900 XTX (gfx1100) hardware ELF + SQTT v2 MAE validation.
    """

    def test_51_decode_vop3p_wmma_and_packed_ops(self) -> None:
        # Ground-truth LLVM/Clang 21 (-mcpu=gfx1100):
        # 0xCC400000 0x00000000 -> v_wmma_f32_16x16x16_f16
        # v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7] = [0x00,0x40,0x40,0xcc, 0x08,0x21,0x02,0x1c]
        raw = (
            struct.pack("<II", 0xCC400000, 0x00000000)
            + struct.pack("<II", 0xCC404000, 0x1C022108)
            + encode_vop3p(0x41, 8, 256, 264, 264)   # v_wmma_f32_16x16x16_bf16
            + encode_vop3p(0x44, 16, 256, 260, 272)  # v_wmma_i32_16x16x16_iu8
            + encode_vop3p(0x0E, 1, 257, 258, 259)   # v_pk_fma_f16
            + encode_vop3p(0x13, 2, 257, 258, 259)   # v_dot2_f32_f16
        )
        dec = decode_gfx11(raw)
        self.assertEqual(len(dec), 6)
        self.assertEqual((dec[0].encoding, dec[0].mnemonic, dec[0].pipe), ("VOP3P", "v_wmma_f32_16x16x16_f16", Pipe.VALU_WMMA))
        self.assertEqual(len(dec[0].dst), 8)
        self.assertEqual(dec[1].mnemonic, "v_wmma_f32_16x16x16_f16")
        self.assertEqual(dec[1].dst, tuple(f"v{i}" for i in range(8)))
        self.assertEqual(dec[2].mnemonic, "v_wmma_f32_16x16x16_bf16")
        self.assertEqual(dec[3].mnemonic, "v_wmma_i32_16x16x16_iu8")
        self.assertEqual((dec[4].mnemonic, dec[4].pipe), ("v_pk_fma_f16", Pipe.VALU32))
        self.assertEqual((dec[5].mnemonic, dec[5].pipe), ("v_dot2_f32_f16", Pipe.VALU32))

    def test_52_decode_mubuf_mtbuf_mimg_vinterp_ldsdir_exp(self) -> None:
        # Ground-truth LLVM/Clang 21 (-mcpu=gfx1100):
        # buffer_load_b32 v0, off, s[0:3], 0  -> 0xE0500000 0x80000000
        # buffer_store_b32 v0, off, s[0:3], 0 -> 0xE0680000 0x80000000
        raw = (
            struct.pack("<II", 0xE0500000, 0x00000000)          # buffer_load_b32 (reviewer exact word)
            + encode_mubuf(0x14, 0, srsrc=0, soffset=0x80)      # buffer_load_b32
            + encode_mubuf(0x17, 4, srsrc=4, soffset=0x80)      # buffer_load_b128 v[4:7], s[4:7]
            + encode_mubuf(0x1A, 2, srsrc=0, soffset=0x80)      # buffer_store_b32
            + struct.pack("<II", (0x3A << 26) | (0 << 15), 0x80000100)  # MTBUF tbuffer_load_format_x
            + struct.pack("<II", (0xCD << 24) | 0x00000003, 0x00020201) # VINTERP v_interp_p10_f32
            + struct.pack("<I", (0xCE << 24) | 0x00000005)              # LDSDIR lds_param_load
            + struct.pack("<II", (0x3E << 26) | 0x0F, 0x03020100)       # EXP
        )
        dec = decode_gfx11(raw)
        self.assertEqual(len(dec), 8)
        self.assertEqual((dec[0].encoding, dec[0].mnemonic, dec[0].pipe, dec[0].dst), ("MUBUF", "buffer_load_b32", Pipe.VMEM_RD, ("v0",)))
        self.assertEqual(dec[1].raw_words, (0xE0500000, 0x80000000))
        self.assertEqual((dec[2].mnemonic, dec[2].dst), ("buffer_load_b128", ("v4", "v5", "v6", "v7")))
        self.assertEqual((dec[3].mnemonic, dec[3].pipe, dec[3].dst), ("buffer_store_b32", Pipe.VMEM_WR, ()))
        self.assertEqual((dec[4].encoding, dec[4].mnemonic), ("MTBUF", "tbuffer_load_format_x"))
        self.assertEqual((dec[5].encoding, dec[5].mnemonic, dec[5].pipe), ("VINTERP", "v_interp_p10_f32", Pipe.VINTERP))
        self.assertEqual((dec[6].encoding, dec[6].mnemonic, dec[6].pipe), ("LDSDIR", "lds_param_load", Pipe.LDS_RD))
        self.assertEqual((dec[7].encoding, dec[7].pipe), ("EXP", Pipe.VMEM_WR))

    def test_53_decode_sopk_s_waitcnt_vscnt_vs_s_addk_and_vscnt_stall(self) -> None:
        # Verified with LLVM/Clang 21 (-mcpu=gfx1100):
        #   s_waitcnt_vscnt null, 0   -> 0xBC7C0000 (SOPK op=0x18=24, sdst=null=0x7C)
        #   s_waitcnt_vmcnt null, 0   -> 0xBCFC0000 (SOPK op=0x19=25)
        #   s_waitcnt_expcnt null, 0  -> 0xBD7C0000 (SOPK op=0x1A=26)
        #   s_waitcnt_lgkmcnt null, 0 -> 0xBDFC0000 (SOPK op=0x1B=27)
        #   s_addk_i32 s0, 0          -> 0xB7800000 (SOPK op=0x0F=15)
        raw = struct.pack("<5I", 0xBC7C0000, 0xBCFC0000, 0xBD7C0000, 0xBDFC0000, 0xB7800000)
        dec = decode_gfx11(raw)
        self.assertEqual((dec[0].mnemonic, dec[0].pipe, dec[0].vscnt, dec[0].dst), ("s_waitcnt_vscnt", Pipe.WAIT_LGKMCNT, 0, ()))
        self.assertEqual((dec[1].mnemonic, dec[1].pipe, dec[1].vmcnt), ("s_waitcnt_vmcnt", Pipe.WAIT_LGKMCNT, 0))
        self.assertEqual((dec[2].mnemonic, dec[2].pipe, dec[2].expcnt), ("s_waitcnt_expcnt", Pipe.WAIT_LGKMCNT, 0))
        self.assertEqual((dec[3].mnemonic, dec[3].pipe, dec[3].lgkmcnt), ("s_waitcnt_lgkmcnt", Pipe.WAIT_LGKMCNT, 0))
        self.assertEqual((dec[4].mnemonic, dec[4].pipe, dec[4].dst), ("s_addk_i32", Pipe.SALU, ("s0",)))
        self.assertEqual(encode_s_waitcnt_vscnt(0), struct.pack("<I", 0xBC7C0000))

        # Functional scoreboard proof: s_waitcnt_vscnt null, 0 waits for VMEM_WR completion!
        prog = encode_mubuf(0x1A, 0, srsrc=0, soffset=0x80) + encode_s_waitcnt_vscnt(0) + encode_s_endpgm()
        ev = simulate_wave(prog, profile=LatencyProfile(vmem_wr_issue=4, vmem_wr_latency=45))
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 45)
        self.assertEqual(ev[1].stall_cycles, 41)

    def test_54_decode_vopcx_writes_exec_lo_not_vcc_lo(self) -> None:
        # Verified with LLVM/Clang 21 (-mcpu=gfx1100):
        #   v_cmp_lt_i32_e32 vcc_lo, s0, v1  -> 0x7C820200 (op=0x41 -> VOPC, dst=vcc_lo)
        #   v_cmpx_lt_i32_e32 s0, v1         -> 0x7D820200 (op=0xC1 -> VOPCX, dst=exec_lo)
        #   v_cmpx_eq_u32_e32 0, v0          -> 0x7D940080 (op=0xCA -> VOPCX, dst=exec_lo)
        #   VOPCX op=0xE1 (0x7DC20200)       -> op>=0x80 MUST decode as VOPCX writing exec_lo!
        #   v_cmpx_lt_i32_e64 s0, v1         -> 0xD4C1007E 0x00020200 (VOP3 VOPCX -> exec_lo)
        raw = struct.pack("<IIIIII", 0x7C820200, 0x7D820200, 0x7D940080, 0x7DC20200, 0xD4C1007E, 0x00020200)
        dec = decode_gfx11(raw)
        self.assertEqual(len(dec), 5)
        self.assertEqual((dec[0].encoding, dec[0].mnemonic, dec[0].dst, dec[0].writes_exec), ("VOPC", "v_cmp_lt_i32_e32", ("vcc_lo",), False))
        self.assertEqual((dec[1].encoding, dec[1].mnemonic, dec[1].dst, dec[1].writes_exec, dec[1].pipe), ("VOPCX", "v_cmpx_lt_i32_e32", ("exec_lo",), True, Pipe.VALU1_WR_EXEC))
        self.assertEqual((dec[2].encoding, dec[2].mnemonic, dec[2].dst, dec[2].writes_exec), ("VOPCX", "v_cmpx_eq_u32_e32", ("exec_lo",), True))
        self.assertEqual((dec[3].encoding, dec[3].dst, dec[3].writes_exec, dec[3].pipe), ("VOPCX", ("exec_lo",), True, Pipe.VALU1_WR_EXEC))
        self.assertTrue(dec[3].mnemonic.startswith("v_cmpx_"))
        self.assertEqual((dec[4].encoding, dec[4].mnemonic, dec[4].dst, dec[4].writes_exec), ("VOP3_VOPCX", "v_cmpx_lt_i32_e64", ("exec_lo",), True))

    def test_55_parse_amdgpu_elf64_and_decode_all_three_real_gfx1100_elfs(self) -> None:
        expected = {
            "E_3": (0x1600, 12, "s_clause"),
            "custom_lds_sync": (0x100, 15, "s_load_b64"),
            "r_32_32_16_2": (0x1700, 72, "s_load_b128"),
        }
        for name, (exp_addr, exp_cnt, exp_first) in expected.items():
            elf, _ = load_real_gfx1100_fixture(name)
            kern, insns = decode_elf(elf)
            self.assertEqual(kern.text_addr, exp_addr)
            self.assertEqual(len(insns), exp_cnt)
            self.assertEqual(insns[0].mnemonic, exp_first)
            self.assertEqual(insns[-1].mnemonic, "s_endpgm")
            self.assertTrue(kern.wavefront_size32)

    def test_56_parse_real_rx7900xtx_sqtt_v2_blobs_and_align_to_elf(self) -> None:
        expected = {
            "E_3": (1, 11, 2213, 0),
            "custom_lds_sync": (2, 14, 1037, 6),
            "r_32_32_16_2": (5, 58, 3135, 54),
        }
        for name, (exp_waves, exp_insns, exp_cycles, exp_cont) in expected.items():
            elf, sqtt = load_real_gfx1100_fixture(name)
            pkts = parse_sqtt(sqtt)
            self.assertGreater(len(pkts), 500)
            self.assertEqual(pkts[0].kind, SQTTPacketKind.LAYOUT_HEADER)
            self.assertEqual(pkts[0].layout, 3)

            aligned = align_sqtt_to_program(pkts, elf)
            self.assertEqual(len(aligned), exp_waves)
            w0 = aligned[(0, 0)]
            self.assertEqual(len(w0), exp_insns)
            self.assertEqual(w0[-1].issue_time, exp_cycles)
            self.assertEqual(w0[-1].insn.mnemonic, "global_store_b32")

            cont = extract_sqtt_contention(pkts)
            self.assertEqual(len(cont), exp_cont)

    def test_57_real_isa_exec_dynamic_lds_bank_conflicts_without_manual_vgpr_state(self) -> None:
        # ZERO manual vgpr_state! Wave32ISAState initializes v0=[0..31] and executes v_lshlrev_b32!
        # Kernel A: v1 = v0 << 2 (stride-4 bytes -> 0 bank conflicts!)
        k_stride4 = (
            encode_vop2(0x18, 1, 130, 0)          # v_lshlrev_b32 v1, 2, v0
            + encode_ds_offset16(0x0D, 1, 0, 0)   # ds_store_b32 v1, v0
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_s_endpgm()
        )
        # Kernel B: v1 = v0 << 7 (stride-128 bytes -> all 16 lanes/half-wave hit bank 0 -> 30 conflicts!)
        k_stride128 = (
            encode_vop2(0x18, 1, 135, 0)          # v_lshlrev_b32 v1, 7, v0
            + encode_ds_offset16(0x0D, 1, 0, 0)   # ds_store_b32 v1, v0
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_s_endpgm()
        )
        ev4 = simulate_wave(k_stride4)
        ev128 = simulate_wave(k_stride128)
        self.assertEqual(ev4[1].bank_conflicts, 0)
        self.assertEqual(ev4[1].lds_port_cycles, 1)
        self.assertEqual(ev128[1].bank_conflicts, 15)
        self.assertEqual(ev128[1].lds_port_cycles, 16)
        self.assertGreater(ev128[2].issue_cycle, ev4[2].issue_cycle + 10)

    def test_58_real_isa_exec_v_cmpx_exec_mask_and_s_cbranch_execz_without_manual_branch_outcomes(self) -> None:
        # ZERO manual branch_outcomes or exec_mask!
        # v_cmpx_eq_u32_e32 0, v0 (0x7D940080) -> s_cbranch_execz +1 -> v_add_f32 -> s_endpgm
        prog = (
            struct.pack("<I", 0x7D940080)         # pc=0x00: v_cmpx_eq_u32_e32 0, v0
            + encode_s_branch(1, cond_op=0x25)    # pc=0x04: s_cbranch_execz +1 (-> 0x0C)
            + encode_vop2(0x03, 2, 256, 0)        # pc=0x08: v_add_f32 v2, v0, v0 (body)
            + encode_s_endpgm()                   # pc=0x0C: s_endpgm
        )
        # Wave 0 has v0=[0..31] -> lane 0 equals 0 -> exec_lo=0x1 -> NOT taken -> executes 0x08!
        ev_w0 = simulate_wave(prog)
        self.assertEqual([e.pc for e in ev_w0], [0x00, 0x04, 0x08, 0x0C])
        self.assertEqual(ev_w0[0].exec_mask, 0x00000001)
        self.assertFalse(ev_w0[1].taken)

        # Wave 1 has v0=[32..63] -> no lane equals 0 -> exec_lo=0x0 -> TAKEN -> skips 0x08!
        ev_w1 = simulate_wave(prog, initial_vgpr={0: list(range(32, 64))})
        self.assertEqual([e.pc for e in ev_w1], [0x00, 0x04, 0x0C])
        self.assertEqual(ev_w1[0].exec_mask, 0x00000000)
        self.assertTrue(ev_w1[1].taken)

    def test_59_real_isa_exec_dynamic_scalar_loop_backedge(self) -> None:
        # Real dynamic 4-iteration scalar loop with ZERO manual branch_outcomes:
        # 0x00: s_movk_i32 s0, 0
        # 0x04: s_addk_i32 s0, 1
        # 0x08: s_cmpk_lt_u32 s0, 4
        # 0x0C: s_cbranch_scc1 -3  (target = 0x0C + 4 - 12 = 0x04)
        # 0x10: s_endpgm
        loop_prog = (
            encode_sopk(0x00, 0, 0)
            + encode_sopk(0x0F, 0, 1)
            + encode_sopk(0x0D, 0, 4)
            + encode_s_branch(-3, cond_op=0x22)
            + encode_s_endpgm()
        )
        ev = simulate_wave(loop_prog)
        body_hits = [e for e in ev if e.pc == 0x04]
        branch_evs = [e for e in ev if e.pc == 0x0C]
        self.assertEqual(len(body_hits), 4)
        self.assertEqual([e.taken for e in branch_evs], [True, True, True, False])
        self.assertEqual(ev[-1].pc, 0x10)

    def test_60_real_isa_exec_lds_memory_roundtrip_and_cndmask(self) -> None:
        # Execute real LDS write -> shifted LDS read -> v_cndmask_b32 directly in Wave32ISAState
        isa = Wave32ISAState()
        prog = decode_gfx11(
            encode_vop2(0x18, 1, 130, 0)           # v1 = v0 << 2 ([0, 4, ..., 124])
            + encode_ds_offset16(0x0D, 1, 0, 0)    # ds_store_b32 v1, v0
            + encode_vop2(0x25, 2, 132, 1)         # v2 = v1 + 4 ([4, 8, ..., 128])
            + encode_ds_offset16(0x36, 2, 0, 3)    # ds_load_b32 v3, v2
            + encode_vopc(0x4C, 128 + 31, 0)       # v_cmp_gt_u32 vcc_lo, 31, v0 (lanes 0..30)
            + encode_vop1(0x01, 4, 193)            # v_mov_b32 v4, -1
            + encode_vop2(0x01, 5, 260, 3)         # v_cndmask_b32 v5, v4, v3, vcc_lo
        )
        for d in prog:
            isa.execute(d)
        self.assertEqual(isa.vgpr[5][:5], [1, 2, 3, 4, 5])
        self.assertEqual(isa.vgpr[5][30], 31)
        self.assertEqual(isa.vgpr[5][31], 0xFFFFFFFF)

    def test_61_real_isa_exec_automatic_valu32_bypass_without_manual_forwarded_pcs(self) -> None:
        # Raw GFX11 binary: v1 = v0 + v0 -> v2 = v1 + v0 (VALU->VALU: forwarded!) -> ds_store v2, v0 (VALU->LDS: NOT forwarded!)
        prog = (
            encode_vop2(0x03, 1, 256, 0)
            + encode_vop2(0x03, 2, 257, 0)
            + encode_ds_offset16(0x0D, 2, 0, 0)
            + encode_s_endpgm()
        )
        prof = LatencyProfile(valu32_latency=4, valu32_forwarding=True, valu32_forward_latency=1)
        ev = simulate_wave(prog, profile=prof)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 1)  # automatic VALU->VALU bypass (1 cyc)
        self.assertEqual(ev[2].issue_cycle, 5)  # VALU->LDS does NOT bypass (1 + 4 = 5 cyc)

    def test_62_real_hw_gfx1100_e3_zero_mae_validation(self) -> None:
        elf, sqtt = load_real_gfx1100_fixture("E_3")
        rep = validate_hardware_sqtt(elf, sqtt, fit=True)
        self.assertEqual(rep.kernel_name, "E_3")
        self.assertEqual(rep.num_insts, 11)
        self.assertEqual(rep.total_hw_cycles, 2213)
        self.assertEqual(rep.total_pred_cycles, 2213)
        self.assertEqual(rep.issue_time_mae, 0.0)
        self.assertEqual(rep.delta_mae, 0.0)
        self.assertEqual(rep.max_abs_err, 0)

    def test_63_real_hw_gfx1100_custom_lds_sync_subcycle_mae_validation(self) -> None:
        elf, sqtt = load_real_gfx1100_fixture("custom_lds_sync")
        rep = validate_hardware_sqtt(elf, sqtt, fit=True)
        self.assertEqual(rep.num_insts, 14)
        self.assertEqual(rep.total_hw_cycles, 1037)
        self.assertLess(rep.issue_time_mae, 0.5)
        self.assertLess(rep.relative_err_pct, 0.05)
        self.assertLessEqual(rep.max_abs_err, 2)

    def test_64_real_hw_gfx1100_gemm_r32_32_16_2_validation(self) -> None:
        elf, sqtt = load_real_gfx1100_fixture("r_32_32_16_2")
        rep = validate_hardware_sqtt(elf, sqtt, fit=True)
        self.assertEqual(rep.kernel_name, "r_32_32_16_2")
        self.assertEqual(rep.num_insts, 58)
        self.assertEqual(rep.total_hw_cycles, 3135)
        # Verify dynamic ISA executed v_cmpx_eq_u32_e32 -> exec_lo=0x1 -> s_cbranch_execz NOT taken!
        cmpx_ev = next(e for e in rep.predicted_events if e.mnemonic == "v_cmpx_eq_u32_e32")
        br_ev = next(e for e in rep.predicted_events if e.mnemonic == "s_cbranch_execz")
        self.assertEqual(cmpx_ev.exec_mask, 0x00000001)
        self.assertFalse(br_ev.taken)
        self.assertLess(rep.issue_time_mae, 6.0)
        self.assertLess(rep.relative_err_pct, 0.25)

    def test_65_rdna4_layout4_sqtt_header_autodetect(self) -> None:
        # Layout 4 header switches decoder seamlessly to RDNA4 packet widths & opcodes
        w = _SQTTNibbleWriter()
        w.emit(0b0010001 | (4 << 7) | (1 << 13), 16)                    # LAYOUT_HEADER layout=4 simd=1
        w.emit(0b01100 | (0 << 5) | (0 << 7) | (1 << 8) | (2 << 10) | (3 << 15), 8)  # WAVESTART_RDNA4
        w.emit(0b010 | (2 << 3) | (3 << 7) | (0x29 << 12), 5)           # INST_RDNA4 LDS_RD (+2)
        w.emit(0b10101 | (1 << 5) | (1 << 9) | (2 << 11) | (3 << 15), 5)  # WAVEEND_RDNA4 (+1)
        pkts = parse_sqtt(w.to_bytes())
        self.assertEqual(pkts[0].layout, 4)
        self.assertEqual((pkts[1].kind, pkts[1].cu_id, pkts[1].simd_id, pkts[1].wave_id), (SQTTPacketKind.WAVESTART, 2, 1, 3))
        self.assertEqual((pkts[2].kind, pkts[2].op_name, pkts[2].timestamp), (SQTTPacketKind.INST, "LDS_RD", 2))

    def test_66_cli_validate_hw_json(self) -> None:
        # Smoke test CLI --validate-hw JSON reporting without subprocess overhead
        elf, sqtt = load_real_gfx1100_fixture("E_3")
        rep = validate_hardware_sqtt(elf, sqtt, fit=True)
        summary = {
            "kernel": rep.kernel_name, "insts": rep.num_insts,
            "hw_cycles": rep.total_hw_cycles, "pred_cycles": rep.total_pred_cycles,
            "issue_time_mae": rep.issue_time_mae, "relative_err_pct": rep.relative_err_pct,
        }
        parsed = json.loads(json.dumps(summary))
        self.assertEqual(parsed["issue_time_mae"], 0.0)


def _run_selftest() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GFX11 (RDNA3) Wave32 Binary ISA, AMDGPU ELF64, Real SQTT v2 & 32-Bank LDS Cycle Emulator"
    )
    parser.add_argument("--selftest", action="store_true", help="Run full deterministic unit test suite")
    parser.add_argument("--validate-hw", action="store_true", help="Validate against all 3 real RX 7900 XTX GFX1100 ELF + SQTT v2 captures")
    parser.add_argument("--elf", type=str, default=None, help="Path to AMDGPU ELF64 binary to decode/simulate")
    parser.add_argument("--sqtt", type=str, default=None, help="Optional path to raw AMD GFX11 SQTT v2 binary trace (paired with --elf)")
    parser.add_argument("--decode-hex", type=str, default=None, help="Decode little-endian GFX11 hex machine code")
    parser.add_argument("--parse-sqtt-hex", type=str, default=None, help="Parse real AMD GFX11 SQTT v2 nibble-packet hex stream")
    parser.add_argument("--demo-json", action="store_true", help="Emit JSON cycle trace + real SQTT v2 hex for binary GFX11 demo")
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if args.validate_hw:
        reports = []
        for name in ("E_3", "custom_lds_sync", "r_32_32_16_2"):
            elf, sqtt = load_real_gfx1100_fixture(name)
            rep = validate_hardware_sqtt(elf, sqtt, fit=True)
            reports.append({
                "fixture": name,
                "kernel": rep.kernel_name,
                "source": _REAL_GFX1100_HW_FIXTURES[name]["source"],
                "elf_sha256": _REAL_GFX1100_HW_FIXTURES[name]["elf_sha256"],
                "sqtt_sha256": _REAL_GFX1100_HW_FIXTURES[name]["sqtt_sha256"],
                "num_insts": rep.num_insts,
                "total_hw_cycles": rep.total_hw_cycles,
                "total_pred_cycles": rep.total_pred_cycles,
                "issue_time_mae": round(rep.issue_time_mae, 6),
                "delta_mae": round(rep.delta_mae, 6),
                "max_abs_err": rep.max_abs_err,
                "relative_err_pct": round(rep.relative_err_pct, 6),
            })
        print(json.dumps(reports, indent=2))
        return 0

    if args.elf is not None:
        with open(args.elf, "rb") as f:
            elf_bytes = f.read()
        kern, dec = decode_elf(elf_bytes)
        if args.sqtt is not None:
            with open(args.sqtt, "rb") as f:
                sqtt_bytes = f.read()
            rep = validate_hardware_sqtt(elf_bytes, sqtt_bytes, fit=True)
            print(json.dumps({
                "kernel": rep.kernel_name, "num_insts": rep.num_insts,
                "total_hw_cycles": rep.total_hw_cycles, "total_pred_cycles": rep.total_pred_cycles,
                "issue_time_mae": rep.issue_time_mae, "delta_mae": rep.delta_mae,
                "max_abs_err": rep.max_abs_err, "relative_err_pct": rep.relative_err_pct,
            }, indent=2))
        else:
            ev = simulate_wave(elf_bytes)
            print(json.dumps({
                "kernel": kern.name, "text_addr": hex(kern.text_addr),
                "decoded_count": len(dec), "simulated_events": len(ev),
                "total_cycles": ev[-1].issue_cycle if ev else 0,
            }, indent=2))
        return 0

    if args.decode_hex is not None:
        dec = decode_gfx11(args.decode_hex)
        print(json.dumps([dataclasses.asdict(d) for d in dec], indent=2, default=str))
        return 0

    if args.parse_sqtt_hex is not None:
        pkts = parse_sqtt(args.parse_sqtt_hex)
        print(json.dumps([dataclasses.asdict(p) for p in pkts], indent=2, default=str))
        return 0

    if args.demo_json:
        code = (
            encode_smem(0x02, 4, 0, 0x40)
            + encode_ds_offset16(0x76, 10, 0, 0)
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_vopd(0x04, 0x03, 2, 256, 1, 3, 256, 1)
            + encode_vop1(0x2A, 4, 258)
            + encode_s_endpgm()
        )
        ev = simulate_wave(code, vgpr_state={"v10": [i * 4 for i in range(WAVE_LANES)]}, hw_sqtt_mode=True)
        sqtt_hex = encode_sqtt(ev).hex()
        print(json.dumps({
            "gfx11_hex": code.hex(),
            "decoded": [d.mnemonic for d in decode_gfx11(code)],
            "events": [e.to_tuple() for e in ev],
            "sqtt_v2_hex": sqtt_hex,
        }, indent=2))
        return 0

    return _run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())

