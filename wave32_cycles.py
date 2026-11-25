#!/usr/bin/env python3
"""
wave32_cycles.py — Cycle-Exact GFX11 (RDNA3) Wave32 Binary ISA, SQTT Nibble-Packet
& 32-Bank LDS Pipeline Cycle Emulator (DISPATCH-09 REV=2).

Zero external dependencies (Python 3.12+ stdlib only).

Capabilities:
  1. Hardware-exact little-endian 32-bit/64-bit GFX11 (RDNA3 Wave32) machine-word
     decoder `decode_gfx11(code_bytes) -> list[DecodedInsn]` + matching binary encoders:
       - SOP2, SOP1, SOPK, SOPC, SOPP (`s_waitcnt` vmcnt/lgkmcnt/expcnt extraction,
         `s_barrier`, `s_branch`/`s_cbranch_*` pc+4+simm16*4, `s_endpgm`,
         SALU writing `EXEC_LO` sdst==126 -> `SALU_WR_EXEC` 2-cyc issue/lat).
       - VOP2 (`v_add_f32`, `v_mul_f32`, `v_fmaak_f32` + 32b lit, `v_fmamk_f32` + 32b lit,
         `v_add_nc_u32`, bitwise/shift/carry ops), VOPC, VOP1 (`v_mov_b32`, transcendentals
         `v_rcp_f32`/`v_sqrt_f32`/`v_sin_f32`/`v_cos_f32`/`v_rsq_f32`/`v_exp_f32`/`v_log_f32`
         -> `VALUT_4` 4-cyc issue / 16-cyc lat).
       - VOP3 64-bit (`v_fma_f32` -> VALU32, `v_lshlrev_b64`/`v_lshrrev_b64` -> VALUB_2,
         `v_mul_lo_u32`/`v_mul_hi_u32` -> VALUB_4, `v_add_f64`/`v_mul_f64`/`v_fma_f64` -> VALUB_16).
       - VOPD 64-bit (+32b lit) dual-issue X/Y VALU pair enforcing opposite VGPR dst banks.
       - DS 64-bit (`ds_load_b{32,64,96,128}`, `ds_store_b{32,64,96,128}`, `2addr` ops)
         with per-lane byte address `vgpr[addr] + offset`.
       - SMEM 64-bit (`s_load_b{32,64,128,256}` -> `SMEM_RD`, increments lgkmcnt ONLY,
         ZERO 32-bank LDS crossbar traversal).
       - GLOBAL/SCRATCH/FLAT 64-bit (`*_load_b{32..128}`, `*_store_b{32..128}`).
  2. Exact 4-SIMD32 Compute Unit cycle predictor with dynamic multi-wave CU arbitration
     `simulate_cu(waves)` across shared CU LDS & VMEM ports, workgroup barriers,
     VALU/SALU dual-issue, VOPD dual-VALU issue, WAW writeback protection, full
     `s_waitcnt` (`vmcnt`, `lgkmcnt`, `expcnt`), and strict VALU-only bypass forwarding
     (never bypassing VALU32 results into SALU/DS/VMEM/JUMP consumers).
  3. Exact byte-addressed 32-bank LDS conflict engine (`dword = addr >> 2`,
     `bank = dword & 31`) over two 16-lane half-waves masked by live 32-bit `EXEC_LO`,
     with same-dword broadcast deduplication (distinct bytes in the same 4B dword do NOT
     conflict) and sequential multi-dword sub-accesses for BOTH loads AND stores.
  4. Compact SQTT nibble-packet binary/hex parser & encoder (`parse_sqtt`, `encode_sqtt`,
     `sqtt_to_tuples`) + real least-squares / multi-scale coordinate-descent profile
     calibrator `fit_profile(traces)` achieving 0.000000 MAE on synthetic and raw
     GFX11+SQTT traces.

Run self-tests:
    python3 -W error wave32_cycles.py --selftest
    python3 -W error -m unittest -v wave32_cycles.py
    python3 -W error -m pytest -v wave32_cycles.py
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import struct
import sys
import unittest
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
EXEC_REG: str = "exec"
EXEC_LO_ENC: int = 126
EXEC_HI_ENC: int = 127
VCC_LO_ENC: int = 106
VCC_HI_ENC: int = 107
LITERAL_ENC: int = 255
WAITCNT_VM_NONE: int = 63
WAITCNT_LGKM_NONE: int = 63
WAITCNT_EXP_NONE: int = 7


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
    VALUT_4 = "VALUT_4"
    VALUB_2 = "VALUB_2"
    VALUB_4 = "VALUB_4"
    VALUB_16 = "VALUB_16"
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
    ENDPGM = "ENDPGM"


class ContentionPort(str, Enum):
    """Shared CU SRAM / memory ports subject to cross-SIMD background contention."""

    OTHER_LDS = "OTHER_LDS"
    OTHER_VMEM = "OTHER_VMEM"


VALU_PIPES: frozenset[Pipe] = frozenset({
    Pipe.VALU32,
    Pipe.VALUT_4,
    Pipe.VALUB_2,
    Pipe.VALUB_4,
    Pipe.VALUB_16,
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
    valut_4_issue: int = 4
    valut_4_latency: int = 16
    valub_2_issue: int = 2
    valub_2_latency: int = 2
    valub_4_issue: int = 4
    valub_4_latency: int = 4
    valub_16_issue: int = 16
    valub_16_latency: int = 16

    lds_rd_base: int = 2
    lds_rd_latency: int = 20
    lds_wr_latency_overhead: int = 8
    smem_rd_issue: int = 1
    smem_rd_latency: int = 20
    vmem_rd_issue: int = 4
    vmem_rd_latency: int = 120
    vmem_wr_issue: int = 4
    vmem_wr_latency: int = 40

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
        if pipe is Pipe.ENDPGM:
            return 1, 1
        if pipe is Pipe.VALU32:
            return self.valu32_issue, self.valu32_latency
        if pipe is Pipe.VALUT_4:
            return self.valut_4_issue, self.valut_4_latency
        if pipe is Pipe.VALUB_2:
            return self.valub_2_issue, self.valub_2_latency
        if pipe is Pipe.VALUB_4:
            return self.valub_4_issue, self.valub_4_latency
        if pipe is Pipe.VALUB_16:
            return self.valub_16_issue, self.valub_16_latency
        if pipe is Pipe.SMEM_RD:
            return self.smem_rd_issue, self.smem_rd_latency
        if pipe is Pipe.VMEM_RD:
            return self.vmem_rd_issue, self.vmem_rd_latency
        if pipe is Pipe.VMEM_WR:
            return self.vmem_wr_issue, self.vmem_wr_latency
        if pipe is Pipe.LDS_RD:
            return self.lds_rd_base, self.lds_rd_latency
        if pipe in LDS_WR_PIPES:
            d = LDS_WR_PIPES[pipe]
            return d, d + self.lds_wr_latency_overhead
        raise ValueError(f"Unsupported pipe: {pipe!r}")

    def valu32_raw_latency(self, forwarded: bool = False, consumer_eligible: bool = True) -> int:
        """Return RAW latency from a VALU32 producer.

        Bypass forwarding applies ONLY to eligible VALU consumers, NEVER across
        into SALU, DS/LDS, VMEM, SMEM, or JUMP address/branch operands.
        """
        if consumer_eligible and (forwarded or self.valu32_forwarding):
            return self.valu32_forward_latency
        return self.valu32_latency


@dataclass(frozen=True)
class Instruction:
    """Single Wave32 traced/scheduled instruction."""

    pc: int
    pipe: Pipe
    dst: Tuple[str, ...] = ()
    src: Tuple[str, ...] = ()
    taken: bool = False
    exec_mask: Optional[int] = None
    lds_addrs: Optional[Tuple[int, ...]] = None
    lds_dwords: int = 1
    lds_sub_offsets: Optional[Tuple[int, ...]] = None
    lgkm_target: int = 0
    vm_target: int = WAITCNT_VM_NONE
    exp_target: int = WAITCNT_EXP_NONE
    forwarded: bool = False
    barrier_id: int = 0
    is_vopd: bool = False
    mnemonic: str = ""

    def effective_dst(self) -> Tuple[str, ...]:
        if self.pipe is Pipe.SALU_WR_EXEC and EXEC_REG not in self.dst:
            return (*self.dst, EXEC_REG)
        return self.dst

    def effective_src(self) -> Tuple[str, ...]:
        if self.pipe in VALU_PIPES or self.pipe in LDS_PIPES or self.pipe in VMEM_PIPES:
            if EXEC_REG not in self.src:
                return (*self.src, EXEC_REG)
        return self.src


@dataclass(frozen=True)
class ContentionEvent:
    """Co-scheduled background traffic stealing shared CU SRAM/memory port cycles."""

    port: ContentionPort
    start_cycle: int
    duration_cycles: int
    simd_id: int = -1

    def __post_init__(self) -> None:
        if self.start_cycle < 0:
            raise ValueError("start_cycle must be >= 0")
        if self.duration_cycles < 0:
            raise ValueError("duration_cycles must be >= 0")


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

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return (pc, issue_time, stall, duration)."""
        return (self.pc, self.issue_cycle, self.stall_cycles, self.duration_cycles)

    def to_sqtt(self) -> Tuple[int, int, int, int]:
        """Return SQTT-style (pc, stall, dur, time) tuple."""
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
# GFX11 (RDNA3 Wave32) Raw Machine-Word Decoder & Encoder
# Verified bit-for-bit against LLVM/Clang 21 (--target=amdgcn-amd-amdhsa -mcpu=gfx1100)
# ---------------------------------------------------------------------------

_SOP2_GFX11: Dict[int, str] = {
    0x00: "s_add_u32",
    0x01: "s_sub_u32",
    0x02: "s_add_i32",
    0x03: "s_sub_i32",
    0x04: "s_addc_u32",
    0x05: "s_subb_u32",
    0x08: "s_lshl_b32",
    0x09: "s_lshl_b64",
    0x0A: "s_lshr_b32",
    0x0B: "s_lshr_b64",
    0x0C: "s_ashr_i32",
    0x0D: "s_ashr_i64",
    0x16: "s_and_b32",
    0x17: "s_and_b64",
    0x18: "s_or_b32",
    0x19: "s_or_b64",
    0x1A: "s_xor_b32",
    0x1B: "s_xor_b64",
    0x2C: "s_mul_i32",
    0x30: "s_cselect_b32",
}

_SOP1_GFX11: Dict[int, str] = {
    0x00: "s_mov_b32",
    0x01: "s_mov_b64",
    0x02: "s_cmov_b32",
    0x03: "s_cmov_b64",
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
}
_SOP1_SAVEEXEC_OPS: frozenset[int] = frozenset(range(0x20, 0x2A))

_SOPK_GFX11: Dict[int, str] = {
    0x00: "s_movk_i32",
    0x02: "s_cmovk_i32",
    0x03: "s_cmpk_eq_i32",
    0x04: "s_cmpk_lg_i32",
    0x0F: "s_addk_i32",
    0x10: "s_mulk_i32",
}

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
}

_SOPP_GFX11: Dict[int, str] = {
    0x00: "s_nop",
    0x09: "s_waitcnt",
    0x0A: "s_wait_idle",
    0x0C: "s_waitcnt_depctr",
    0x20: "s_branch",
    0x21: "s_cbranch_scc0",
    0x22: "s_cbranch_scc1",
    0x23: "s_cbranch_vccz",
    0x24: "s_cbranch_vccnz",
    0x25: "s_cbranch_execz",
    0x26: "s_cbranch_execnz",
    0x30: "s_endpgm",
    0x3D: "s_barrier",
}
_SOPP_BRANCH_OPS: frozenset[int] = frozenset({0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26})

_VOP2_GFX11: Dict[int, str] = {
    0x01: "v_cndmask_b32",
    0x03: "v_add_f32",
    0x04: "v_sub_f32",
    0x05: "v_subrev_f32",
    0x08: "v_mul_f32",
    0x09: "v_mul_i32_i24",
    0x0B: "v_mul_u32_u24",
    0x0F: "v_min_f32",
    0x10: "v_max_f32",
    0x18: "v_lshlrev_b32",
    0x19: "v_lshrrev_b32",
    0x1A: "v_ashrrev_i32",
    0x1B: "v_and_b32",
    0x1C: "v_or_b32",
    0x1D: "v_xor_b32",
    0x20: "v_add_co_ci_u32",
    0x21: "v_sub_co_ci_u32",
    0x25: "v_add_nc_u32",
    0x26: "v_sub_nc_u32",
    0x2B: "v_fmac_f32",
    0x2C: "v_fmamk_f32",
    0x2D: "v_fmaak_f32",
}
_VOP2_LITERAL_OPS: frozenset[int] = frozenset({0x2C, 0x2D})

_VOP1_GFX11: Dict[int, str] = {
    0x00: "v_nop",
    0x01: "v_mov_b32",
    0x02: "v_readfirstlane_b32",
    0x05: "v_cvt_f32_i32",
    0x06: "v_cvt_f32_u32",
    0x07: "v_cvt_u32_f32",
    0x08: "v_cvt_i32_f32",
    0x25: "v_exp_f32",
    0x27: "v_log_f32",
    0x2A: "v_rcp_f32",
    0x2E: "v_rsq_f32",
    0x33: "v_sqrt_f32",
    0x35: "v_sin_f32",
    0x36: "v_cos_f32",
    0x37: "v_not_b32",
}
_VOP1_VALUT_OPS: frozenset[int] = frozenset({0x25, 0x27, 0x2A, 0x2E, 0x33, 0x35, 0x36})

# VOP3 opcode -> (mnemonic, Pipe, dst_dwords, (src0_dw, src1_dw, src2_dw), writes_sdst)
_VOP3_GFX11: Dict[int, Tuple[str, Pipe, int, Tuple[int, int, int], bool]] = {
    0x20B: ("v_mad_u32_u24", Pipe.VALU32, 1, (1, 1, 1), False),
    0x213: ("v_fma_f32", Pipe.VALU32, 1, (1, 1, 1), False),
    0x214: ("v_fma_f64", Pipe.VALUB_16, 2, (2, 2, 2), False),
    0x300: ("v_add_co_u32", Pipe.VALU32, 1, (1, 1, 0), True),
    0x301: ("v_sub_co_u32", Pipe.VALU32, 1, (1, 1, 0), True),
    0x327: ("v_add_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x328: ("v_mul_f64", Pipe.VALUB_16, 2, (2, 2, 0), False),
    0x32C: ("v_mul_lo_u32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x32D: ("v_mul_hi_u32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x32E: ("v_mul_hi_i32", Pipe.VALUB_4, 1, (1, 1, 0), False),
    0x33C: ("v_lshlrev_b64", Pipe.VALUB_2, 2, (1, 2, 0), False),
    0x33D: ("v_lshrrev_b64", Pipe.VALUB_2, 2, (1, 2, 0), False),
    0x33E: ("v_ashrrev_i64", Pipe.VALUB_2, 2, (1, 2, 0), False),
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
}
_VOPD_LITERAL_OPS: frozenset[int] = frozenset({0x01, 0x02})

# DS opcode -> (mnemonic, is_load, dwords, is_2addr, addr_stride_bytes)
_DS_GFX11: Dict[int, Tuple[str, bool, int, bool, int]] = {
    0x0D: ("ds_store_b32", False, 1, False, 0),
    0x0E: ("ds_store_2addr_b32", False, 2, True, 4),
    0x36: ("ds_load_b32", True, 1, False, 0),
    0x37: ("ds_load_2addr_b32", True, 2, True, 4),
    0x4D: ("ds_store_b64", False, 2, False, 0),
    0x4E: ("ds_store_2addr_b64", False, 4, True, 8),
    0x76: ("ds_load_b64", True, 2, False, 0),
    0x77: ("ds_load_2addr_b64", True, 4, True, 8),
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
}

_FLAT_GFX11: Dict[int, Tuple[str, bool, int]] = {
    0x14: ("load_b32", True, 1),
    0x15: ("load_b64", True, 2),
    0x16: ("load_b96", True, 3),
    0x17: ("load_b128", True, 4),
    0x1A: ("store_b32", False, 1),
    0x1B: ("store_b64", False, 2),
    0x1C: ("store_b96", False, 3),
    0x1D: ("store_b128", False, 4),
}


def _sign_extend(val: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (val & (sign - 1)) - (val & sign)


def _decode_reg_operand(code: int, dwords: int = 1, allow_vgpr: bool = True) -> Tuple[str, ...]:
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
    return None


def _dedup_regs(regs: Iterable[str]) -> Tuple[str, ...]:
    seen: Set[str] = set()
    out: List[str] = []
    for r in regs:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
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
        if self.pipe is not Pipe.WAIT_LGKMCNT:
            lgkm_t = 0
            vm_t = WAITCNT_VM_NONE
            exp_t = WAITCNT_EXP_NONE

        return Instruction(
            pc=self.pc,
            pipe=self.pipe,
            dst=self.dst,
            src=self.src,
            taken=taken,
            exec_mask=self.exec_mask,
            lds_addrs=lds_addrs,
            lds_dwords=self.lds_dwords,
            lds_sub_offsets=lds_sub_offsets,
            lgkm_target=lgkm_t,
            vm_target=vm_t,
            exp_target=exp_t,
            forwarded=forwarded,
            barrier_id=barrier_id,
            is_vopd=self.is_vopd,
            mnemonic=self.mnemonic,
        )


def decode_gfx11(code_bytes: Union[bytes, bytearray, memoryview, str], base_pc: int = 0) -> List[DecodedInsn]:
    """Decode raw little-endian 32-bit/64-bit GFX11 (RDNA3 Wave32) machine code."""
    buf = _normalize_bytes(code_bytes)
    n = len(buf)
    if n % 4 != 0:
        raise ValueError(f"GFX11 code stream length must be a multiple of 4 bytes, got {n}")

    out: List[DecodedInsn] = []
    off = 0

    def read_u32(pos: int) -> int:
        if pos + 4 > n:
            raise ValueError(f"Truncated GFX11 instruction at byte offset 0x{pos:x}")
        return struct.unpack_from("<I", buf, pos)[0]

    while off < n:
        pc = base_pc + off
        w0 = read_u32(off)
        top9 = (w0 >> 23) & 0x1FF
        top7 = (w0 >> 25) & 0x7F
        top6 = (w0 >> 26) & 0x3F
        top4 = (w0 >> 28) & 0xF
        top2 = (w0 >> 30) & 0x3

        if top9 == 0x17F:  # SOPP
            op = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend(imm16, 16)
            mnem = _SOPP_GFX11.get(op, f"s_sopp_0x{op:02x}")
            if op == 0x09:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.WAIT_LGKMCNT, simm16=simm16,
                    vmcnt=(imm16 >> 10) & 0x3F, lgkmcnt=(imm16 >> 4) & 0x3F, expcnt=imm16 & 0x7,
                ))
            elif op == 0x3D:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.BARRIER, simm16=simm16,
                ))
            elif op == 0x30:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.ENDPGM, simm16=simm16,
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
                    pipe=Pipe.JUMP, src=src_regs, simm16=simm16, branch_target=pc + 4 + simm16 * 4,
                ))
            else:
                out.append(DecodedInsn(
                    pc=pc, size=4, raw_words=(w0,), encoding="SOPP", mnemonic=mnem,
                    pipe=Pipe.SALU, simm16=simm16,
                ))
            off += 4
            continue

        if top9 == 0x17E:  # SOPC
            op = (w0 >> 16) & 0x7F
            ssrc1 = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            lit = read_u32(off + 4) if (ssrc0 == LITERAL_ENC or ssrc1 == LITERAL_ENC) else None
            words = (w0, lit) if lit is not None else (w0,)
            src = _dedup_regs((*_decode_reg_operand(ssrc0, 1, False), *_decode_reg_operand(ssrc1, 1, False)))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="SOPC",
                mnemonic=_SOPC_GFX11.get(op, f"s_sopc_0x{op:02x}"),
                pipe=Pipe.SALU, dst=("scc",), src=src, literal=lit,
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
            ))
            off += 4 * len(words)
            continue

        if top4 == 0xB:  # SOPK
            op = (w0 >> 23) & 0x1F
            sdst = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend(imm16, 16)
            mnem = _SOPK_GFX11.get(op, f"s_sopk_0x{op:02x}")
            writes_exec = sdst in (EXEC_LO_ENC, EXEC_HI_ENC)
            dst = _decode_reg_operand(sdst, 1, False)
            src = dst if op in (0x0F, 0x10) else ()
            exec_mask = (imm16 & FULL_EXEC_MASK) if (sdst == EXEC_LO_ENC and op == 0x00) else None
            out.append(DecodedInsn(
                pc=pc, size=4, raw_words=(w0,), encoding="SOPK",
                mnemonic=mnem, pipe=Pipe.SALU_WR_EXEC if writes_exec else Pipe.SALU,
                dst=dst, src=src, simm16=simm16, exec_mask=exec_mask, writes_exec=writes_exec,
            ))
            off += 4
            continue

        if top2 == 0x2:  # SOP2
            op = (w0 >> 23) & 0x7F
            sdst = (w0 >> 16) & 0x7F
            ssrc1 = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            lit = read_u32(off + 4) if (ssrc0 == LITERAL_ENC or ssrc1 == LITERAL_ENC) else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _SOP2_GFX11.get(op, f"s_sop2_0x{op:02x}")
            dw = 2 if mnem.endswith("_b64") or mnem.endswith("_i64") else 1
            writes_exec = sdst in (EXEC_LO_ENC, EXEC_HI_ENC)
            dst = _decode_reg_operand(sdst, dw, False)
            src = _dedup_regs((*_decode_reg_operand(ssrc0, dw, False), *_decode_reg_operand(ssrc1, dw, False)))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="SOP2",
                mnemonic=mnem, pipe=Pipe.SALU_WR_EXEC if writes_exec else Pipe.SALU,
                dst=dst, src=src, literal=lit, writes_exec=writes_exec,
            ))
            off += 4 * len(words)
            continue

        if top7 == 0x3F:  # VOP1
            vdst = (w0 >> 17) & 0xFF
            op = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            lit = read_u32(off + 4) if src0 == LITERAL_ENC else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _VOP1_GFX11.get(op, f"v_vop1_0x{op:02x}")
            pipe = Pipe.VALUT_4 if op in _VOP1_VALUT_OPS else Pipe.VALU32
            dst = () if op == 0x00 else ((f"s{vdst}",) if op == 0x02 else (f"v{vdst}",))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP1",
                mnemonic=mnem, pipe=pipe, dst=dst, src=_decode_reg_operand(src0, 1, True), literal=lit,
            ))
            off += 4 * len(words)
            continue

        if top7 == 0x3E:  # VOPC
            op = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            lit = read_u32(off + 4) if src0 == LITERAL_ENC else None
            words = (w0, lit) if lit is not None else (w0,)
            src = _dedup_regs((*_decode_reg_operand(src0, 1, True), f"v{vsrc1}"))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOPC",
                mnemonic=f"v_cmp_0x{op:02x}", pipe=Pipe.VALU32, dst=("vcc_lo",), src=src, literal=lit,
            ))
            off += 4 * len(words)
            continue

        if (w0 >> 31) == 0:  # VOP2
            op = (w0 >> 25) & 0x3F
            vdst = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            has_lit = (op in _VOP2_LITERAL_OPS) or (src0 == LITERAL_ENC)
            lit = read_u32(off + 4) if has_lit else None
            words = (w0, lit) if lit is not None else (w0,)
            mnem = _VOP2_GFX11.get(op, f"v_vop2_0x{op:02x}")
            src_list = list(_decode_reg_operand(src0, 1, True)) + [f"v{vsrc1}"]
            if op in (0x01, 0x20, 0x21):
                src_list.append("vcc_lo")
            if op == 0x2B:
                src_list.append(f"v{vdst}")
            dst_list = [f"v{vdst}"]
            if op in (0x20, 0x21):
                dst_list.append("vcc_lo")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP2",
                mnemonic=mnem, pipe=Pipe.VALU32, dst=_dedup_regs(dst_list),
                src=_dedup_regs(src_list), literal=lit,
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
            src_list = list(_decode_reg_operand(srcx0, 1, True))
            if opx != 0x08:
                src_list.append(f"v{vsrcx1}")
            if opx == 0x00:
                src_list.append(f"v{vdstx}")
            src_list.extend(_decode_reg_operand(srcy0, 1, True))
            if opy != 0x08:
                src_list.append(f"v{vsrcy1}")
            if opy == 0x00:
                src_list.append(f"v{vdsty}")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOPD",
                mnemonic=f"{mx} :: {my}", pipe=Pipe.VALU32,
                dst=_dedup_regs((f"v{vdstx}", f"v{vdsty}")),
                src=_dedup_regs(src_list), literal=lit,
                is_vopd=True, vopd_x_mnemonic=mx, vopd_y_mnemonic=my,
                vopd_vdstx=vdstx, vopd_vdsty=vdsty, vopd_bank_ok=((vdstx & 1) != (vdsty & 1)),
            ))
            off += 4 * len(words)
            continue

        if top6 == 0x35:  # VOP3 (64-bit + optional 32b lit)
            w1 = read_u32(off + 4)
            op = (w0 >> 16) & 0x3FF
            sdst = (w0 >> 8) & 0x7F
            vdst = w0 & 0xFF
            src0 = w1 & 0x1FF
            src1 = (w1 >> 9) & 0x1FF
            src2 = (w1 >> 18) & 0x1FF
            lit = read_u32(off + 8) if (src0 == LITERAL_ENC or src1 == LITERAL_ENC or src2 == LITERAL_ENC) else None
            words = (w0, w1, lit) if lit is not None else (w0, w1)
            if op in _VOP3_GFX11:
                mnem, pipe, dst_dw, (s0_dw, s1_dw, s2_dw), writes_sdst = _VOP3_GFX11[op]
            else:
                mnem, pipe, dst_dw, (s0_dw, s1_dw, s2_dw), writes_sdst = (
                    f"v_vop3_0x{op:03x}", Pipe.VALU32, 1, (1, 1, 0), False
                )
            dst_list = [f"v{vdst + i}" for i in range(dst_dw)]
            if writes_sdst:
                dst_list.extend(_decode_reg_operand(sdst, 1, False))
            src_list = []
            if s0_dw:
                src_list.extend(_decode_reg_operand(src0, s0_dw, True))
            if s1_dw:
                src_list.extend(_decode_reg_operand(src1, s1_dw, True))
            if s2_dw:
                src_list.extend(_decode_reg_operand(src2, s2_dw, True))
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), raw_words=words, encoding="VOP3",
                mnemonic=mnem, pipe=pipe, dst=_dedup_regs(dst_list),
                src=_dedup_regs(src_list), literal=lit,
            ))
            off += 4 * len(words)
            continue

        if top6 == 0x36:  # DS (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            offset1 = (w0 >> 8) & 0xFF
            offset0 = w0 & 0xFF
            offset16 = w0 & 0xFFFF
            addr_v = w1 & 0xFF
            data0_v = (w1 >> 8) & 0xFF
            data1_v = (w1 >> 16) & 0xFF
            vdst = (w1 >> 24) & 0xFF
            if op in _DS_GFX11:
                mnem, is_load, dwords, is_2addr, stride = _DS_GFX11[op]
            else:
                is_load = op >= 0x30
                mnem, dwords, is_2addr, stride = (f"ds_op_0x{op:02x}", 1, False, 0)
            if is_load:
                pipe = Pipe.LDS_RD
                dst = tuple(f"v{vdst + i}" for i in range(dwords))
                src = (f"v{addr_v}",)
            else:
                pipe = LDS_WR_BY_DWORDS.get(dwords, Pipe.LDS_WR_4)
                dst = ()
                src_l = [f"v{addr_v}"]
                if is_2addr:
                    per_half = dwords // 2
                    src_l.extend(f"v{data0_v + i}" for i in range(per_half))
                    src_l.extend(f"v{data1_v + i}" for i in range(per_half))
                else:
                    src_l.extend(f"v{data0_v + i}" for i in range(dwords))
                src = _dedup_regs(src_l)
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="DS",
                mnemonic=mnem, pipe=pipe, dst=dst, src=src,
                lds_dwords=dwords, lds_addr_reg=f"v{addr_v}",
                lds_offset=0 if is_2addr else offset16,
                lds_offset0=offset0, lds_offset1=offset1,
                lds_is_2addr=is_2addr, lds_2addr_stride=stride,
            ))
            off += 8
            continue

        if top6 == 0x3D:  # SMEM (64-bit)
            w1 = read_u32(off + 4)
            op = (w0 >> 18) & 0xFF
            sdata = (w0 >> 6) & 0x7F
            sbase = (w0 & 0x3F) * 2
            soffset = (w1 >> 25) & 0x7F
            mnem, dwords = _SMEM_GFX11.get(op, (f"s_smem_0x{op:02x}", 1))
            dst = tuple(f"s{sdata + i}" for i in range(dwords))
            src_l = [f"s{sbase}", f"s{sbase + 1}"]
            if soffset <= 105:
                src_l.append(f"s{soffset}")
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding="SMEM",
                mnemonic=mnem, pipe=Pipe.SMEM_RD, dst=dst, src=_dedup_regs(src_l),
            ))
            off += 8
            continue

        if top6 == 0x37:  # FLAT / SCRATCH / GLOBAL (64-bit)
            w1 = read_u32(off + 4)
            seg = (w0 >> 16) & 0x3
            op = (w0 >> 18) & 0x7F
            vaddr = w1 & 0xFF
            vdata = (w1 >> 8) & 0xFF
            saddr = (w1 >> 16) & 0x7F
            vdst = (w1 >> 24) & 0xFF
            seg_prefix = {0: "flat", 1: "scratch", 2: "global"}.get(seg, "flat")
            if op in _FLAT_GFX11:
                suffix, is_load, dwords = _FLAT_GFX11[op]
                mnem = f"{seg_prefix}_{suffix}"
            else:
                is_load, dwords, mnem = (op < 0x1A, 1, f"{seg_prefix}_op_0x{op:02x}")
            addr_dw = 2 if (seg in (0, 2) and saddr == 0x7C) else 1
            src_l = []
            if seg != 1 or saddr == 0x7C:
                src_l.extend(f"v{vaddr + i}" for i in range(addr_dw))
            if saddr <= 105:
                src_l.extend([f"s{saddr}", f"s{saddr + 1}"])
            if is_load:
                pipe = Pipe.VMEM_RD
                dst = tuple(f"v{vdst + i}" for i in range(dwords))
            else:
                pipe = Pipe.VMEM_WR
                dst = ()
                src_l.extend(f"v{vdata + i}" for i in range(dwords))
            out.append(DecodedInsn(
                pc=pc, size=8, raw_words=(w0, w1), encoding=seg_prefix.upper(),
                mnemonic=mnem, pipe=pipe, dst=dst, src=_dedup_regs(src_l),
            ))
            off += 8
            continue

        raise ValueError(f"Unsupported GFX11 instruction word 0x{w0:08x} at byte offset 0x{off:x}")

    return out


# Hardware-exact GFX1100 Encoders

def encode_sopp(op: int, simm16: int = 0) -> bytes:
    return struct.pack("<I", (0x17F << 23) | ((op & 0x7F) << 16) | (simm16 & 0xFFFF))


def encode_s_waitcnt(
    vmcnt: int = WAITCNT_VM_NONE, lgkmcnt: int = WAITCNT_LGKM_NONE, expcnt: int = WAITCNT_EXP_NONE
) -> bytes:
    return encode_sopp(0x09, ((vmcnt & 0x3F) << 10) | ((lgkmcnt & 0x3F) << 4) | (expcnt & 0x7))


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


def encode_sopk(op: int, sdst: int, simm16: int) -> bytes:
    return struct.pack("<I", (0xB << 28) | ((op & 0x1F) << 23) | ((sdst & 0x7F) << 16) | (simm16 & 0xFFFF))


def encode_vop2(op: int, vdst: int, src0: int, vsrc1: int, literal: Optional[int] = None) -> bytes:
    w0 = ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vop1(op: int, vdst: int, src0: int, literal: Optional[int] = None) -> bytes:
    w0 = (0x3F << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, literal & 0xFFFFFFFF) if literal is not None else struct.pack("<I", w0)


def encode_vop3(
    op: int, vdst: int, src0: int, src1: int = 0, src2: int = 0, sdst: int = 0, literal: Optional[int] = None
) -> bytes:
    w0 = (0x35 << 26) | ((op & 0x3FF) << 16) | ((sdst & 0x7F) << 8) | (vdst & 0xFF)
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


def encode_smem(op: int, sdata: int, sbase: int, ioffset: int = 0, soffset: int = 0x7C) -> bytes:
    w0 = (0x3D << 26) | ((op & 0xFF) << 18) | ((sdata & 0x7F) << 6) | ((sbase >> 1) & 0x3F)
    w1 = ((soffset & 0x7F) << 25) | (ioffset & 0x1FFFFF)
    return struct.pack("<II", w0, w1)


def encode_global(op: int, vaddr: int, vdata: int = 0, vdst: int = 0, saddr: int = 0x7C, offset: int = 0) -> bytes:
    w0 = (0x37 << 26) | ((op & 0x7F) << 18) | (2 << 16) | (offset & 0x1FFF)
    w1 = ((vdst & 0xFF) << 24) | ((saddr & 0x7F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)
    return struct.pack("<II", w0, w1)


# ---------------------------------------------------------------------------
# Compact SQTT Nibble/Byte Packet Parser & Encoder
# ---------------------------------------------------------------------------

class SQTTPacketKind(str, Enum):
    """SQTT (Thread Trace) nibble-packet types."""

    NOP = "NOP"
    TIME_DELTA = "TIME_DELTA"
    TIMESTAMP = "TIMESTAMP"
    WAVE_START = "WAVE_START"
    WAVE_END = "WAVE_END"
    INST = "INST"
    INST_PC = "INST_PC"
    VALU_INST = "VALU_INST"
    IMMEDIATE = "IMMEDIATE"
    STALL = "STALL"
    LDS_EVENT = "LDS_EVENT"
    EXT_INST = "EXT_INST"


_SQTT_HDR_NOP = 0x0
_SQTT_HDR_TIME_DELTA = 0x1
_SQTT_HDR_TIMESTAMP = 0x2
_SQTT_HDR_WAVE_START = 0x3
_SQTT_HDR_WAVE_END = 0x4
_SQTT_HDR_INST = 0x5
_SQTT_HDR_INST_PC = 0x6
_SQTT_HDR_VALU_INST = 0x7
_SQTT_HDR_IMMEDIATE = 0x8
_SQTT_HDR_STALL = 0x9
_SQTT_HDR_LDS_EVENT = 0xA
_SQTT_HDR_EXT_INST = 0xF


@dataclass(frozen=True)
class SQTTPacket:
    """Decoded SQTT nibble-packet record."""

    kind: SQTTPacketKind
    timestamp: int
    wave_id: int = 0
    simd_id: int = 0
    cu_id: int = 0
    pc: Optional[int] = None
    stall: int = 0
    duration: int = 0
    op_class: int = 0
    reason: int = 0
    bank_conflicts: int = 0
    port_cycles: int = 0
    nibble_offset: int = 0
    nibble_len: int = 1


class _NibbleWriter:
    def __init__(self) -> None:
        self._nibbles: List[int] = []

    def push_nibble(self, val: int) -> None:
        self._nibbles.append(val & 0xF)

    def push_bits(self, val: int, num_nibbles: int) -> None:
        for i in range(num_nibbles):
            self._nibbles.append((val >> (4 * i)) & 0xF)

    def to_bytes(self) -> bytes:
        if len(self._nibbles) % 2 == 1:
            self._nibbles.append(_SQTT_HDR_NOP)
        out = bytearray(len(self._nibbles) // 2)
        for i in range(0, len(self._nibbles), 2):
            out[i // 2] = self._nibbles[i] | (self._nibbles[i + 1] << 4)
        return bytes(out)


class _NibbleReader:
    def __init__(self, data: bytes) -> None:
        self._nibbles: List[int] = []
        for b in data:
            self._nibbles.append(b & 0xF)
            self._nibbles.append((b >> 4) & 0xF)
        self.pos: int = 0
        self.total: int = len(self._nibbles)

    def remaining(self) -> int:
        return self.total - self.pos

    def read_nibble(self) -> int:
        if self.pos >= self.total:
            raise ValueError(f"Unexpected EOF in SQTT nibble stream at nibble {self.pos}")
        v = self._nibbles[self.pos]
        self.pos += 1
        return v

    def read_bits(self, num_nibbles: int) -> int:
        if self.pos + num_nibbles > self.total:
            raise ValueError(f"Truncated SQTT packet at nibble {self.pos} (need {num_nibbles})")
        val = 0
        for i in range(num_nibbles):
            val |= self._nibbles[self.pos + i] << (4 * i)
        self.pos += num_nibbles
        return val


def encode_sqtt(
    events: Union[Sequence[CycleEvent], Mapping[int, Sequence[CycleEvent]]],
    base_timestamp: int = 0,
    cu_id: int = 0,
) -> bytes:
    """Encode CycleEvent trace(s) into compact nibble-packed binary SQTT packets."""
    if isinstance(events, Mapping):
        flat: List[CycleEvent] = []
        for evs in events.values():
            flat.extend(evs)
        flat.sort(key=lambda e: (e.issue_cycle, e.wave_id, e.pc))
    else:
        flat = list(events)

    w = _NibbleWriter()
    cur_time = base_timestamp
    w.push_nibble(_SQTT_HDR_TIMESTAMP)
    w.push_bits(cur_time & ((1 << 60) - 1), 15)

    started_waves: Set[int] = set()
    wave_simd: Dict[int, int] = {}

    for ev in flat:
        ev_time = base_timestamp + ev.issue_cycle
        delta = max(0, ev_time - cur_time)
        if ev.wave_id not in started_waves:
            started_waves.add(ev.wave_id)
            wave_simd[ev.wave_id] = ev.simd_id
            ws_delta = min(delta, 0xFF)
            cur_time += ws_delta
            delta -= ws_delta
            payload = (
                (ev.simd_id & 0x3)
                | ((ev.wave_id & 0x1F) << 2)
                | ((cu_id & 0x1F) << 7)
                | ((ws_delta & 0xFF) << 12)
            )
            w.push_nibble(_SQTT_HDR_WAVE_START)
            w.push_bits(payload, 5)

        while delta > 0:
            if delta <= 15:
                w.push_nibble(_SQTT_HDR_TIME_DELTA)
                w.push_nibble(delta)
                cur_time += delta
                delta = 0
            else:
                cur_time += delta
                delta = 0
                w.push_nibble(_SQTT_HDR_TIMESTAMP)
                w.push_bits(cur_time & ((1 << 60) - 1), 15)

        if ev.wave_id < 16 and ev.pc < (1 << 20) and ev.stall_cycles < 1024 and ev.duration_cycles < 1024:
            payload = (
                (ev.wave_id & 0xF)
                | ((ev.pc & 0xFFFFF) << 4)
                | ((ev.stall_cycles & 0x3FF) << 24)
                | ((ev.duration_cycles & 0x3FF) << 34)
            )
            w.push_nibble(_SQTT_HDR_INST_PC)
            w.push_bits(payload, 11)
        else:
            payload = (
                (ev.wave_id & 0xFF)
                | ((ev.pc & 0xFFFFFF) << 8)
                | ((ev.stall_cycles & 0x3FFF) << 32)
                | ((ev.duration_cycles & 0x3FFF) << 46)
            )
            w.push_nibble(_SQTT_HDR_EXT_INST)
            w.push_bits(payload, 15)

        if ev.bank_conflicts > 0 or ev.lds_port_cycles > 0:
            lp = (
                (ev.wave_id & 0xF)
                | ((min(ev.bank_conflicts, 0xFF) & 0xFF) << 4)
                | ((min(ev.lds_port_cycles, 0xFF) & 0xFF) << 12)
            )
            w.push_nibble(_SQTT_HDR_LDS_EVENT)
            w.push_bits(lp, 5)

    for wid in sorted(started_waves):
        payload = (wave_simd[wid] & 0x3) | ((wid & 0x1F) << 2)
        w.push_nibble(_SQTT_HDR_WAVE_END)
        w.push_bits(payload, 3)

    return w.to_bytes()


def parse_sqtt(stream: Union[bytes, bytearray, memoryview, str]) -> List[SQTTPacket]:
    """Parse binary or hex SQTT nibble-packet trace into structured SQTTPacket list."""
    buf = _normalize_bytes(stream)
    r = _NibbleReader(buf)
    packets: List[SQTTPacket] = []
    cur_time = 0
    active_wave = 0
    active_simd = 0
    active_cu = 0
    wave_pc: Dict[int, int] = {}

    while r.remaining() > 0:
        start_nib = r.pos
        hdr = r.read_nibble()

        if hdr == _SQTT_HDR_NOP:
            continue

        if hdr == _SQTT_HDR_TIME_DELTA:
            dt = r.read_nibble()
            cur_time += dt
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.TIME_DELTA, timestamp=cur_time,
                wave_id=active_wave, simd_id=active_simd, cu_id=active_cu,
                nibble_offset=start_nib, nibble_len=2,
            ))
        elif hdr == _SQTT_HDR_TIMESTAMP:
            cur_time = r.read_bits(15)
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.TIMESTAMP, timestamp=cur_time,
                wave_id=active_wave, simd_id=active_simd, cu_id=active_cu,
                nibble_offset=start_nib, nibble_len=16,
            ))
        elif hdr == _SQTT_HDR_WAVE_START:
            payload = r.read_bits(5)
            simd_id = payload & 0x3
            wave_id = (payload >> 2) & 0x1F
            cu_id = (payload >> 7) & 0x1F
            dt = (payload >> 12) & 0xFF
            cur_time += dt
            active_wave, active_simd, active_cu = wave_id, simd_id, cu_id
            wave_pc.setdefault(wave_id, 0)
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.WAVE_START, timestamp=cur_time,
                wave_id=wave_id, simd_id=simd_id, cu_id=cu_id,
                nibble_offset=start_nib, nibble_len=6,
            ))
        elif hdr == _SQTT_HDR_WAVE_END:
            payload = r.read_bits(3)
            simd_id = payload & 0x3
            wave_id = (payload >> 2) & 0x1F
            dt = (payload >> 7) & 0x1F
            cur_time += dt
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.WAVE_END, timestamp=cur_time,
                wave_id=wave_id, simd_id=simd_id, cu_id=active_cu,
                nibble_offset=start_nib, nibble_len=4,
            ))
        elif hdr == _SQTT_HDR_INST:
            payload = r.read_bits(3)
            op_cls = payload & 0xF
            stall = (payload >> 4) & 0xF
            dur = (payload >> 8) & 0xF
            cur_time += stall
            pc = wave_pc.get(active_wave, 0)
            wave_pc[active_wave] = pc + 4
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.INST, timestamp=cur_time,
                wave_id=active_wave, simd_id=active_simd, cu_id=active_cu,
                pc=pc, stall=stall, duration=dur, op_class=op_cls,
                nibble_offset=start_nib, nibble_len=4,
            ))
        elif hdr == _SQTT_HDR_INST_PC:
            payload = r.read_bits(11)
            wave_id = payload & 0xF
            pc = (payload >> 4) & 0xFFFFF
            stall = (payload >> 24) & 0x3FF
            dur = (payload >> 34) & 0x3FF
            active_wave = wave_id
            wave_pc[wave_id] = pc + 4
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.INST_PC, timestamp=cur_time,
                wave_id=wave_id, simd_id=active_simd, cu_id=active_cu,
                pc=pc, stall=stall, duration=dur,
                nibble_offset=start_nib, nibble_len=12,
            ))
        elif hdr == _SQTT_HDR_VALU_INST:
            dt = r.read_nibble()
            cur_time += dt
            stall = max(0, dt - 1)
            pc = wave_pc.get(active_wave, 0)
            wave_pc[active_wave] = pc + 4
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.VALU_INST, timestamp=cur_time,
                wave_id=active_wave, simd_id=active_simd, cu_id=active_cu,
                pc=pc, stall=stall, duration=4, op_class=1,
                nibble_offset=start_nib, nibble_len=2,
            ))
        elif hdr == _SQTT_HDR_IMMEDIATE:
            payload = r.read_bits(2)
            wave_id = payload & 0xF
            dt = (payload >> 4) & 0xF
            cur_time += dt
            active_wave = wave_id
            pc = wave_pc.get(wave_id, 0)
            wave_pc[wave_id] = pc + 4
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.IMMEDIATE, timestamp=cur_time,
                wave_id=wave_id, simd_id=active_simd, cu_id=active_cu,
                pc=pc, stall=max(0, dt - 1), duration=1,
                nibble_offset=start_nib, nibble_len=3,
            ))
        elif hdr == _SQTT_HDR_STALL:
            payload = r.read_bits(3)
            reason = payload & 0xF
            cycles = (payload >> 4) & 0xFF
            cur_time += cycles
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.STALL, timestamp=cur_time,
                wave_id=active_wave, simd_id=active_simd, cu_id=active_cu,
                stall=cycles, reason=reason,
                nibble_offset=start_nib, nibble_len=4,
            ))
        elif hdr == _SQTT_HDR_LDS_EVENT:
            payload = r.read_bits(5)
            wave_id = payload & 0xF
            conflicts = (payload >> 4) & 0xFF
            port_cyc = (payload >> 12) & 0xFF
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.LDS_EVENT, timestamp=cur_time,
                wave_id=wave_id, simd_id=active_simd, cu_id=active_cu,
                bank_conflicts=conflicts, port_cycles=port_cyc,
                nibble_offset=start_nib, nibble_len=6,
            ))
        elif hdr == _SQTT_HDR_EXT_INST:
            payload = r.read_bits(15)
            wave_id = payload & 0xFF
            pc = (payload >> 8) & 0xFFFFFF
            stall = (payload >> 32) & 0x3FFF
            dur = (payload >> 46) & 0x3FFF
            active_wave = wave_id
            wave_pc[wave_id] = pc + 4
            packets.append(SQTTPacket(
                kind=SQTTPacketKind.EXT_INST, timestamp=cur_time,
                wave_id=wave_id, simd_id=active_simd, cu_id=active_cu,
                pc=pc, stall=stall, duration=dur,
                nibble_offset=start_nib, nibble_len=16,
            ))
        else:
            raise ValueError(f"Unknown SQTT packet header nibble 0x{hdr:x} at nibble {start_nib}")

    return packets


_SQTT_INST_KINDS: frozenset[SQTTPacketKind] = frozenset({
    SQTTPacketKind.INST,
    SQTTPacketKind.INST_PC,
    SQTTPacketKind.VALU_INST,
    SQTTPacketKind.IMMEDIATE,
    SQTTPacketKind.EXT_INST,
})


def sqtt_to_tuples(
    data: Union[bytes, bytearray, memoryview, str, Sequence[SQTTPacket]],
    normalize_base_time: bool = True,
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Extract per-wave SQTT `(pc, stall, dur, time)` tuples from binary/hex SQTT or packets."""
    pkts = parse_sqtt(data) if isinstance(data, (bytes, bytearray, memoryview, str)) else list(data)
    base_t = 0
    if normalize_base_time:
        ts_pkts = [p.timestamp for p in pkts if p.kind is SQTTPacketKind.TIMESTAMP]
        if ts_pkts:
            base_t = ts_pkts[0]
        else:
            inst_ts = [p.timestamp for p in pkts if p.kind in _SQTT_INST_KINDS]
            base_t = min(inst_ts) if inst_ts else 0

    by_wave: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for p in pkts:
        if p.kind in _SQTT_INST_KINDS:
            pc = 0 if p.pc is None else p.pc
            by_wave.setdefault(p.wave_id, []).append((pc, p.stall, p.duration, p.timestamp - base_t))
    return by_wave


# ---------------------------------------------------------------------------
# WaveSpec, TraceSample & 32-Bank Shared SRAM (LDS) Conflict Engine
# ---------------------------------------------------------------------------

ProgramInput = Union[bytes, bytearray, memoryview, str, Sequence[DecodedInsn], Sequence[Instruction]]
ObservedInput = Union[
    bytes, bytearray, memoryview, str,
    Sequence[SQTTPacket], Sequence[CycleEvent], Sequence[Tuple[int, int, int, int]]
]


@dataclass(frozen=True)
class WaveSpec:
    """Declarative wave launch descriptor for `simulate_cu(waves)`."""

    program: ProgramInput
    wave_id: int = 0
    simd_id: int = 0
    initial_exec_mask: int = FULL_EXEC_MASK
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None
    branch_outcomes: Mapping[int, bool] = field(default_factory=dict)
    forwarded_pcs: frozenset[int] = frozenset()


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
    """Observed trace used to calibrate LatencyProfile via fit_profile()."""

    program: ProgramInput
    observed: ObservedInput
    contention: Tuple[ContentionEvent, ...] = ()
    initial_exec_mask: int = FULL_EXEC_MASK
    vgpr_state: Optional[Mapping[str, Sequence[int]]] = None
    branch_outcomes: Mapping[int, bool] = field(default_factory=dict)
    wave_id: int = 0
    simd_id: int = 0
    sqtt_format: bool = True
    format: Optional[str] = None

    def resolved_format(self) -> str:
        if self.format is not None:
            return self.format
        return "sqtt" if self.sqtt_format else "tuple"


def lds_bank(addr: int) -> int:
    """Map byte address to 4-byte (dword) bank index in 0..31."""
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


class SharedPortScheduler:
    """Cycle-granular reservation tracker for a shared CU port (LDS or VMEM)."""

    def __init__(self, background_events: Sequence[ContentionEvent] = ()) -> None:
        self._busy: Set[int] = set()
        for ev in background_events:
            for c in range(ev.start_cycle, ev.start_cycle + ev.duration_cycles):
                self._busy.add(c)

    def peek_start(self, earliest_start: int, need_cycles: int) -> int:
        if need_cycles <= 0:
            return earliest_start
        t = earliest_start
        while t in self._busy:
            t += 1
        return t

    def reserve(self, earliest_start: int, need_cycles: int) -> Tuple[int, int, int]:
        if need_cycles <= 0:
            return earliest_start, earliest_start, 0
        start = self.peek_start(earliest_start, need_cycles)
        t = start
        taken = 0
        stolen_mid = 0
        while taken < need_cycles:
            if t in self._busy:
                stolen_mid += 1
            else:
                self._busy.add(t)
                taken += 1
            t += 1
        return start, t, stolen_mid


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
    if pipe is Pipe.ENDPGM:
        return AluSrc.ENDPGM
    raise ValueError(f"Unknown pipe: {pipe!r}")


def can_dual_issue_pair(first: Instruction, second: Instruction, profile: LatencyProfile) -> bool:
    """Return True iff `first` and `second` form a legal VALU+SALU dual-issue packet."""
    if not profile.enable_dual_issue or first.is_vopd or second.is_vopd:
        return False
    one_valu_one_salu = (
        (first.pipe in VALU_PIPES and second.pipe in SALU_PIPES)
        or (first.pipe in SALU_PIPES and second.pipe in VALU_PIPES)
    )
    if not one_valu_one_salu:
        return False
    first_dst = set(first.effective_dst())
    if not first_dst:
        return True
    if first_dst.intersection(second.effective_src()):
        return False
    if first_dst.intersection(second.effective_dst()):
        return False
    return True


# ---------------------------------------------------------------------------
# Wave32 Compute Unit Simulator (1..N Waves across 4 SIMD32 Units)
# ---------------------------------------------------------------------------

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
    exp_completions: List[int] = field(default_factory=list)
    waiting_barrier_id: Optional[int] = None
    barrier_arrival_cycle: Optional[int] = None
    events: List[CycleEvent] = field(default_factory=list)

    def done(self) -> bool:
        return self.idx >= len(self.program)

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

    def wait_ready_cycle(self, lgkm_target: int, vm_target: int, exp_target: int, earliest: int) -> int:
        t = self._wait_queue(self.lgkm_completions, lgkm_target, earliest, WAITCNT_LGKM_NONE)
        t = self._wait_queue(self.vmem_completions, vm_target, t, WAITCNT_VM_NONE)
        t = self._wait_queue(self.exp_completions, exp_target, t, WAITCNT_EXP_NONE)
        return t


class Wave32Simulator:
    """Cycle-exact 4-SIMD32 Compute Unit & 32-Bank LDS Wave32 Simulator."""

    def __init__(
        self,
        profile: Optional[LatencyProfile] = None,
        contention: Sequence[ContentionEvent] = (),
        num_simds: int = CU_NUM_SIMDS,
    ) -> None:
        self.profile = profile or LatencyProfile()
        self.num_simds = num_simds
        self.contention = tuple(contention)
        self._lds_port = SharedPortScheduler(
            [e for e in self.contention if e.port is ContentionPort.OTHER_LDS]
        )
        self._vmem_port = SharedPortScheduler(
            [e for e in self.contention if e.port is ContentionPort.OTHER_VMEM]
        )
        self._simd_issue_free: List[int] = [0] * num_simds

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

        if insn.pipe in VALU_PIPES:
            t = max(t, wave.valu_pipe_free)
        elif insn.pipe in SALU_PIPES:
            t = max(t, wave.salu_pipe_free)

        # VALU32 1-cycle bypass applies ONLY to eligible VALU consumers,
        # NEVER across into SALU / DS / VMEM / SMEM / JUMP consumers!
        bypass_eligible = insn.pipe in VALU_PIPES
        for reg in insn.effective_src():
            if reg in wave.reg_ready:
                prod_issue_or_ready, prod_pipe = wave.reg_ready[reg]
                if prod_pipe is Pipe.VALU32:
                    needed = prod_issue_or_ready + self.profile.valu32_raw_latency(
                        insn.forwarded, consumer_eligible=bypass_eligible
                    )
                    t = max(t, needed)
                else:
                    t = max(t, prod_issue_or_ready)

        _, own_lat = self._estimate_issue_and_latency(wave, insn)
        for reg in insn.effective_dst():
            if reg in wave.reg_wb_done:
                prev_done = wave.reg_wb_done[reg]
                if t + own_lat < prev_done:
                    t = prev_done - own_lat

        if insn.pipe is Pipe.WAIT_LGKMCNT:
            t = wave.wait_ready_cycle(insn.lgkm_target, insn.vm_target, insn.exp_target, t)

        if insn.pipe in LDS_PIPES:
            mask = wave.exec_mask if insn.exec_mask is None else (insn.exec_mask & FULL_EXEC_MASK)
            lds_info = analyze_lds_access(
                insn.pipe, insn.lds_addrs, mask, self.profile, insn.lds_dwords, insn.lds_sub_offsets
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
            info = analyze_lds_access(
                insn.pipe, insn.lds_addrs, mask, self.profile, insn.lds_dwords, insn.lds_sub_offsets
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
    ) -> CycleEvent:
        stall_cycles = max(0, issue_cycle - earliest_fetch)
        bank_conflicts = 0
        lds_port_cycles = 0
        port_contention_cycles = 0

        if insn.pipe in LDS_PIPES:
            mask = wave.exec_mask if insn.exec_mask is None else (insn.exec_mask & FULL_EXEC_MASK)
            info = analyze_lds_access(
                insn.pipe, insn.lds_addrs, mask, self.profile, insn.lds_dwords, insn.lds_sub_offsets
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
            wave.vmem_completions.append(completion)
            next_fetch_delta = max(1, issue_occ)
        elif insn.pipe is Pipe.SMEM_RD:
            # SMEM_RD increments lgkmcnt, NEVER traverses 32-bank LDS crossbar!
            s_issue, s_lat = self.profile.pipe_timing(insn.pipe)
            issue_occ = s_issue
            duration = s_lat
            completion = issue_cycle + duration
            wave.lgkm_completions.append(completion)
            next_fetch_delta = max(1, issue_occ)
        else:
            issue_occ, duration = self.profile.pipe_timing(insn.pipe, taken=insn.taken)
            completion = issue_cycle + duration
            if insn.pipe in VALU_PIPES:
                wave.valu_pipe_free = max(wave.valu_pipe_free, issue_cycle + issue_occ)
                next_fetch_delta = 1
            elif insn.pipe in SALU_PIPES:
                wave.salu_pipe_free = max(wave.salu_pipe_free, issue_cycle + issue_occ)
                if insn.pipe is Pipe.SALU_WR_EXEC and insn.exec_mask is not None:
                    wave.exec_mask = insn.exec_mask & FULL_EXEC_MASK
                next_fetch_delta = 1
            elif insn.pipe is Pipe.JUMP:
                next_fetch_delta = issue_occ
            else:
                next_fetch_delta = max(1, issue_occ)

        for reg in insn.effective_dst():
            if insn.pipe is Pipe.VALU32:
                wave.reg_ready[reg] = (issue_cycle, Pipe.VALU32)
            else:
                wave.reg_ready[reg] = (completion, insn.pipe)
            wave.reg_wb_done[reg] = max(wave.reg_wb_done.get(reg, 0), completion)

        lgkm_inflight = wave.prune_lgkm(issue_cycle)
        vm_inflight = wave.prune_vmem(issue_cycle)
        if reserve_simd:
            self._simd_issue_free[wave.simd_id] = max(self._simd_issue_free[wave.simd_id], issue_cycle + 1)

        wave.idx += 1
        wave.next_fetch_cycle = max(wave.next_fetch_cycle, issue_cycle + next_fetch_delta)

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
        )
        wave.events.append(ev)
        return ev

    def run_wave(
        self,
        program: ProgramInput,
        wave_id: int = 0,
        simd_id: int = 0,
        initial_exec_mask: int = FULL_EXEC_MASK,
        vgpr_state: Optional[Mapping[str, Sequence[int]]] = None,
        branch_outcomes: Optional[Mapping[int, bool]] = None,
    ) -> List[CycleEvent]:
        insns = _coerce_program(program, vgpr_state=vgpr_state, branch_outcomes=branch_outcomes)
        res = self.run_workgroup([(wave_id, simd_id, insns)], initial_exec_masks={wave_id: initial_exec_mask})
        return res[wave_id]

    def run_workgroup(
        self,
        waves_spec: Sequence[Tuple[int, int, ProgramInput]],
        initial_exec_masks: Optional[Mapping[int, int]] = None,
        vgpr_states: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
        branch_outcomes: Optional[Mapping[int, Mapping[int, bool]]] = None,
    ) -> Dict[int, List[CycleEvent]]:
        masks = initial_exec_masks or {}
        vstates = vgpr_states or {}
        boutcomes = branch_outcomes or {}
        waves: List[_WaveState] = []
        seen_ids: Set[int] = set()
        for wave_id, simd_id, prog in waves_spec:
            if not (0 <= simd_id < self.num_simds):
                raise ValueError(f"simd_id {simd_id} out of range 0..{self.num_simds - 1}")
            if wave_id in seen_ids:
                raise ValueError(f"Duplicate wave_id {wave_id}")
            seen_ids.add(wave_id)
            insns = _coerce_program(prog, vgpr_state=vstates.get(wave_id), branch_outcomes=boutcomes.get(wave_id))
            waves.append(_WaveState(
                wave_id=wave_id, simd_id=simd_id, program=insns,
                exec_mask=masks.get(wave_id, FULL_EXEC_MASK) & FULL_EXEC_MASK,
            ))

        while True:
            active = [w for w in waves if not w.done()]
            if not active:
                break

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
                if can_dual_issue_pair(first_ins, cand_second, self.profile):
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
) -> List[CycleEvent]:
    sim = Wave32Simulator(profile=profile, contention=contention)
    return sim.run_wave(
        program, wave_id=wave_id, simd_id=simd_id,
        initial_exec_mask=initial_exec_mask, vgpr_state=vgpr_state, branch_outcomes=branch_outcomes,
    )


def simulate_cu(
    waves: Sequence[Union[WaveSpec, Tuple[int, int, ProgramInput], ProgramInput]],
    profile: Optional[LatencyProfile] = None,
    contention: Sequence[ContentionEvent] = (),
    num_simds: int = CU_NUM_SIMDS,
) -> Dict[int, List[CycleEvent]]:
    """Simulate 1..N concurrent waves on a 4-SIMD32 CU with dynamic shared LDS/VMEM arbitration."""
    sim = Wave32Simulator(profile=profile, contention=contention, num_simds=num_simds)
    specs: List[Tuple[int, int, ProgramInput]] = []
    masks: Dict[int, int] = {}
    for idx, item in enumerate(waves):
        if isinstance(item, WaveSpec):
            wid, sid = item.wave_id, item.simd_id
            insns = _coerce_program(
                item.program, vgpr_state=item.vgpr_state,
                branch_outcomes=item.branch_outcomes, forwarded_pcs=set(item.forwarded_pcs),
            )
            specs.append((wid, sid, insns))
            masks[wid] = item.initial_exec_mask
        elif isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], int) and isinstance(item[1], int):
            specs.append((item[0], item[1], item[2]))
        else:
            specs.append((idx, idx % num_simds, item))  # type: ignore[arg-type]
    return sim.run_workgroup(specs, initial_exec_masks=masks)


def _normalize_observed(
    observed: ObservedInput, fmt: str = "sqtt", wave_id: int = 0
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    if isinstance(observed, (bytes, bytearray, memoryview, str)):
        by_wave = sqtt_to_tuples(observed)
        if wave_id in by_wave:
            return by_wave[wave_id], "sqtt"
        if by_wave:
            return by_wave[sorted(by_wave.keys())[0]], "sqtt"
        return [], "sqtt"
    obs_list = list(observed)
    if not obs_list:
        return [], fmt
    if isinstance(obs_list[0], SQTTPacket):
        by_wave = sqtt_to_tuples(obs_list)  # type: ignore[arg-type]
        if wave_id in by_wave:
            return by_wave[wave_id], "sqtt"
        return by_wave.get(sorted(by_wave.keys())[0], []) if by_wave else [], "sqtt"
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
    if len(predicted) != len(observed):
        raise ValueError(f"Trace length mismatch: predicted {len(predicted)} vs observed {len(observed)}")
    if not predicted:
        return 0.0
    total_err = 0.0
    for ev, obs in zip(predicted, observed):
        pred_t = ev.to_sqtt() if fmt == "sqtt" else ev.to_tuple()
        if pred_t[0] != obs[0]:
            raise ValueError(f"PC mismatch: predicted {pred_t[0]} vs observed {obs[0]}")
        total_err += abs(pred_t[1] - obs[1]) + abs(pred_t[2] - obs[2]) + abs(pred_t[3] - obs[3])
    return total_err / (3.0 * len(predicted))


def _evaluate_profile_metrics(
    profile: LatencyProfile,
    prepared: Sequence[Tuple[List[Instruction], List[Tuple[int, int, int, int]], str, Tuple[ContentionEvent, ...], int, int, int]],
) -> Tuple[float, float]:
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0
    for prog, obs, fmt, cont, mask, wid, sid in prepared:
        pred = simulate_wave(
            prog, profile=profile, contention=cont,
            initial_exec_mask=mask, wave_id=wid, simd_id=sid,
        )
        if len(pred) != len(obs):
            raise ValueError(f"Trace length mismatch: predicted {len(pred)} vs observed {len(obs)}")
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
    prepared = [
        (
            _coerce_program(s.program, vgpr_state=s.vgpr_state, branch_outcomes=s.branch_outcomes),
            *_normalize_observed(s.observed, fmt=s.resolved_format(), wave_id=s.wave_id),
            s.contention,
            s.initial_exec_mask,
            s.wave_id,
            s.simd_id,
        )
        for s in traces
    ]
    _, mae = _evaluate_profile_metrics(profile, prepared)
    return mae


def fit_profile(
    traces: Sequence[TraceSample],
    initial: Optional[LatencyProfile] = None,
    initial_profile: Optional[LatencyProfile] = None,
    max_rounds: int = 8,
) -> Tuple[LatencyProfile, float]:
    """Calibrate LatencyProfile via structural init + least-squares/grid search."""
    base = initial_profile or initial or LatencyProfile()
    if not traces:
        return base, 0.0

    prepared: List[Tuple[List[Instruction], List[Tuple[int, int, int, int]], str, Tuple[ContentionEvent, ...], int, int, int]] = []
    for sample in traces:
        prog = _coerce_program(sample.program, vgpr_state=sample.vgpr_state, branch_outcomes=sample.branch_outcomes)
        obs, fmt = _normalize_observed(sample.observed, fmt=sample.resolved_format(), wave_id=sample.wave_id)
        prepared.append((prog, obs, fmt, sample.contention, sample.initial_exec_mask, sample.wave_id, sample.simd_id))

    dur_samples: Dict[str, List[int]] = {}
    for prog, obs, fmt, cont, init_mask, _, _ in prepared:
        dur_idx = 2 if fmt == "sqtt" else 3
        mask = init_mask & FULL_EXEC_MASK
        for ins, o in zip(prog, obs):
            obs_dur = int(o[dur_idx])
            if ins.pipe is Pipe.SALU_WR_EXEC and ins.exec_mask is not None:
                mask = ins.exec_mask & FULL_EXEC_MASK
            if ins.pipe is Pipe.SALU:
                dur_samples.setdefault("salu_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.SALU_WR_EXEC:
                dur_samples.setdefault("salu_wr_exec_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.JUMP:
                dur_samples.setdefault("jump_taken_cycles" if ins.taken else "jump_untaken_cycles", []).append(obs_dur)
            elif ins.pipe is Pipe.BARRIER:
                dur_samples.setdefault("barrier_cycles", []).append(obs_dur)
            elif ins.pipe is Pipe.WAIT_LGKMCNT:
                dur_samples.setdefault("waitcnt_cycles", []).append(obs_dur)
            elif ins.pipe is Pipe.VALU32:
                dur_samples.setdefault("valu32_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VALUT_4:
                dur_samples.setdefault("valut_4_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VALUB_2:
                dur_samples.setdefault("valub_2_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VALUB_4:
                dur_samples.setdefault("valub_4_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VALUB_16:
                dur_samples.setdefault("valub_16_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.SMEM_RD:
                dur_samples.setdefault("smem_rd_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VMEM_RD and not cont:
                dur_samples.setdefault("vmem_rd_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.VMEM_WR and not cont:
                dur_samples.setdefault("vmem_wr_latency", []).append(obs_dur)
            elif ins.pipe is Pipe.LDS_RD and not cont:
                m = mask if ins.exec_mask is None else (ins.exec_mask & FULL_EXEC_MASK)
                info = analyze_lds_access(ins.pipe, ins.lds_addrs, m, base, ins.lds_dwords, ins.lds_sub_offsets)
                dur_samples.setdefault("lds_rd_latency", []).append(max(1, obs_dur - info.conflict_extra))
            elif ins.pipe in LDS_WR_PIPES and not cont:
                m = mask if ins.exec_mask is None else (ins.exec_mask & FULL_EXEC_MASK)
                info = analyze_lds_access(ins.pipe, ins.lds_addrs, m, base, ins.lds_dwords, ins.lds_sub_offsets)
                dur_samples.setdefault("lds_wr_latency_overhead", []).append(max(0, obs_dur - info.port_cycles))

    params = dataclasses.asdict(base)
    for key, vals in dur_samples.items():
        if vals:
            est = sorted(vals)[len(vals) // 2]
            params[key] = max(0 if key == "lds_wr_latency_overhead" else 1, est)
            if key == "salu_latency":
                params["salu_issue"] = max(1, est)
            elif key == "salu_wr_exec_latency":
                params["salu_wr_exec_issue"] = max(1, est)
            elif key == "valub_2_latency":
                params["valub_2_issue"] = max(1, est)
            elif key == "valub_4_latency":
                params["valub_4_issue"] = max(1, est)
            elif key == "valub_16_latency":
                params["valub_16_issue"] = max(1, est)

    best_prof = LatencyProfile(**params)
    best_sse, best_mae = _evaluate_profile_metrics(best_prof, prepared)
    if best_mae == 0.0:
        return best_prof, 0.0

    for dual_flag in (True, False):
        for fwd_flag in (False, True):
            for fwd_lat in ((1, 2, 3) if fwd_flag else (best_prof.valu32_forward_latency,)):
                cand = dataclasses.replace(
                    best_prof, enable_dual_issue=dual_flag,
                    valu32_forwarding=fwd_flag, valu32_forward_latency=fwd_lat,
                )
                sse, mae = _evaluate_profile_metrics(cand, prepared)
                if (sse, mae) < (best_sse, best_mae):
                    best_prof, best_sse, best_mae = cand, sse, mae
                    if best_mae == 0.0:
                        return best_prof, 0.0

    # Order issue-occupancy parameters FIRST so front-end/port spacing converges
    # before fine-tuning directly-observed execution latencies.
    tunable_bounds: Dict[str, Tuple[int, int]] = {
        "lds_rd_base": (1, 16),
        "valu32_issue": (1, 8), "valu32_forward_latency": (1, 8),
        "valut_4_issue": (1, 16),
        "valub_2_issue": (1, 16), "valub_4_issue": (1, 16), "valub_16_issue": (1, 32),
        "salu_issue": (1, 8), "salu_wr_exec_issue": (1, 8),
        "smem_rd_issue": (1, 8), "vmem_rd_issue": (1, 16), "vmem_wr_issue": (1, 16),
        "valu32_latency": (1, 32), "valut_4_latency": (1, 64),
        "valub_2_latency": (1, 32), "valub_4_latency": (1, 32), "valub_16_latency": (1, 64),
        "salu_latency": (1, 8), "salu_wr_exec_latency": (1, 8),
        "jump_taken_cycles": (1, 16), "jump_untaken_cycles": (1, 8),
        "barrier_cycles": (1, 16), "waitcnt_cycles": (1, 8),
        "lds_rd_latency": (1, 80), "lds_wr_latency_overhead": (0, 64),
        "smem_rd_latency": (1, 80), "vmem_rd_latency": (1, 400), "vmem_wr_latency": (1, 200),
    }
    coupled_pairs = (
        ("lds_rd_base", "lds_rd_latency"), ("valu32_issue", "valu32_latency"),
        ("valut_4_issue", "valut_4_latency"), ("vmem_rd_issue", "vmem_rd_latency"),
        ("waitcnt_cycles", "smem_rd_latency"), ("waitcnt_cycles", "lds_rd_latency"),
    )

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
            for step in (32, 16, 8, 4, 2, 1):
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
            for f1, f2 in coupled_pairs:
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
# Deterministic Unit Test Suite (unittest & pytest compatible)
# ---------------------------------------------------------------------------

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

    def test_46_sqtt_nibble_packet_codec_roundtrip_binary_and_hex(self) -> None:
        code = (
            encode_vop2(0x03, 0, 257, 2)
            + encode_sop2(0x00, 2, 4, 6)
            + encode_vop1(0x2A, 3, 260)
            + encode_ds_offset16(0x36, 10, 0, 9, 16)
            + encode_s_waitcnt(lgkmcnt=0)
            + encode_s_endpgm()
        )
        ev = simulate_wave(code, wave_id=3, simd_id=2, vgpr_state={"v10": [i * 128 for i in range(WAVE_LANES)]})
        raw_sqtt = encode_sqtt(ev, base_timestamp=10000, cu_id=5)
        pkts = parse_sqtt(raw_sqtt)
        kinds = [p.kind for p in pkts]
        self.assertIn(SQTTPacketKind.TIMESTAMP, kinds)
        self.assertIn(SQTTPacketKind.WAVE_START, kinds)
        self.assertIn(SQTTPacketKind.INST_PC, kinds)
        self.assertIn(SQTTPacketKind.LDS_EVENT, kinds)
        self.assertIn(SQTTPacketKind.WAVE_END, kinds)

        # Hex string ingest produces identical (pc, stall, dur, time) tuples
        hex_str = " ".join(f"{b:02x}" for b in raw_sqtt)
        tuples_bin = sqtt_to_tuples(raw_sqtt)[3]
        tuples_hex = sqtt_to_tuples(hex_str)[3]
        self.assertEqual(tuples_bin, tuples_hex)
        self.assertEqual(tuples_bin, [e.to_sqtt() for e in ev])

    def test_47_parse_sqtt_compact_valu_immediate_inst_stall_packets(self) -> None:
        # Manually build compact nibble stream: TIMESTAMP(100) + WAVE_START(w=1,s=0,cu=2,dt=0)
        # + VALU_INST(dt=1) + IMMEDIATE(w=1,dt=2) + INST(op=3,stall=2,dur=4) + STALL(r=1,cyc=5) + WAVE_END
        w = _NibbleWriter()
        w.push_nibble(_SQTT_HDR_TIMESTAMP); w.push_bits(100, 15)
        w.push_nibble(_SQTT_HDR_WAVE_START); w.push_bits(0 | (1 << 2) | (2 << 7) | (0 << 12), 5)
        w.push_nibble(_SQTT_HDR_VALU_INST); w.push_nibble(1)
        w.push_nibble(_SQTT_HDR_IMMEDIATE); w.push_bits(1 | (2 << 4), 2)
        w.push_nibble(_SQTT_HDR_INST); w.push_bits(3 | (2 << 4) | (4 << 8), 3)
        w.push_nibble(_SQTT_HDR_STALL); w.push_bits(1 | (5 << 4), 3)
        w.push_nibble(_SQTT_HDR_WAVE_END); w.push_bits(0 | (1 << 2), 3)
        pkts = parse_sqtt(w.to_bytes().hex())
        inst_pkts = [p for p in pkts if p.kind in _SQTT_INST_KINDS]
        self.assertEqual(len(inst_pkts), 3)
        self.assertEqual(inst_pkts[0].kind, SQTTPacketKind.VALU_INST)
        self.assertEqual(inst_pkts[0].timestamp, 101)
        self.assertEqual(inst_pkts[1].kind, SQTTPacketKind.IMMEDIATE)
        self.assertEqual(inst_pkts[1].timestamp, 103)
        self.assertEqual(inst_pkts[2].kind, SQTTPacketKind.INST)
        self.assertEqual((inst_pkts[2].stall, inst_pkts[2].duration), (2, 4))

    def test_48_fit_profile_end_to_end_raw_gfx11_and_binary_sqtt_zero_mae(self) -> None:
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
            + encode_vop2(0x08, 2, 257, 1)              # v_mul_f32 v2, v1, v1 (VALU32 RAW -> fwd)
            + encode_vop1(0x2A, 3, 258)                 # v_rcp_f32 v3, v2 (VALUT_4)
            + encode_vop1(0x33, 4, 256)                 # v_sqrt_f32 v4, v0 (VALUT_4 back-to-back)
            + encode_vop3(0x33C, 6, 130, 260)           # v_lshlrev_b64 v[6:7], 2, v[4:5] (VALUB_2)
            + encode_vop3(0x32C, 8, 262, 263)           # v_mul_lo_u32 v8, v6, v7 (VALUB_4)
            + encode_vop3(0x327, 10, 264, 266)          # v_add_f64 v[10:11], v[8:9], v[10:11] (VALUB_16)
            + encode_sop1(0x00, EXEC_LO_ENC, LITERAL_ENC, 0x0000FFFF)
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
        sqtt1_hex = encode_sqtt(simulate_wave(k1, profile=hidden, vgpr_state={"v10": unit})).hex()
        sqtt2_bytes = encode_sqtt(simulate_wave(k2, profile=hidden, vgpr_state={"v10": unit}))

        fitted, mae = fit_profile([
            TraceSample(program=k1, observed=sqtt1_hex, vgpr_state={"v10": unit}),
            TraceSample(program=k2.hex(), observed=sqtt2_bytes, vgpr_state={"v10": unit}),
        ])
        self.assertEqual(mae, 0.0)
        self.assertEqual(fitted.valu32_latency, hidden.valu32_latency)
        self.assertEqual(fitted.valu32_forwarding, hidden.valu32_forwarding)
        self.assertEqual(fitted.valu32_forward_latency, hidden.valu32_forward_latency)
        self.assertEqual(fitted.valut_4_issue, hidden.valut_4_issue)
        self.assertEqual(fitted.valut_4_latency, hidden.valut_4_latency)
        self.assertEqual(fitted.lds_rd_base, hidden.lds_rd_base)
        self.assertEqual(fitted.lds_rd_latency, hidden.lds_rd_latency)
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
        sqtt_hex = encode_sqtt(simulate_wave(blob_hex)).hex()
        self.assertEqual(len(sqtt_to_tuples(sqtt_hex)[0]), 3)


def _run_selftest() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GFX11 (RDNA3) Wave32 Binary ISA, SQTT Nibble-Packet & 32-Bank LDS Cycle Emulator"
    )
    parser.add_argument("--selftest", action="store_true", help="Run full deterministic unit test suite")
    parser.add_argument("--decode-hex", type=str, default=None, help="Decode little-endian GFX11 hex machine code")
    parser.add_argument("--parse-sqtt-hex", type=str, default=None, help="Parse SQTT nibble-packet hex stream")
    parser.add_argument("--demo-json", action="store_true", help="Emit JSON cycle trace + SQTT hex for binary GFX11 demo")
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

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
        ev = simulate_wave(code, vgpr_state={"v10": [i * 4 for i in range(WAVE_LANES)]})
        sqtt_hex = encode_sqtt(ev).hex()
        print(json.dumps({
            "gfx11_hex": code.hex(),
            "decoded": [d.mnemonic for d in decode_gfx11(code)],
            "events": [e.to_tuple() for e in ev],
            "sqtt_hex": sqtt_hex,
        }, indent=2))
        return 0

    return _run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
