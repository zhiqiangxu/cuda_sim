#!/usr/bin/env python3
"""PTX to C++ translator for cuda_sim.

Translates NVIDIA PTX assembly into C++ code that can run on CPU,
with a thread simulation wrapper (grid/block/thread loop).

Usage:
    python ptx2cpp.py kernel.ptx -o kernel_cpu.cpp
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Register:
    """A PTX register: type prefix + index (e.g. %r3, %rd5, %f1, %p0)."""
    prefix: str  # "r", "rd", "f", "fd", "p"
    index: int

    def cpp_name(self) -> str:
        return f"{self.prefix}[{self.index}]"


@dataclass
class Param:
    """A kernel parameter."""
    name: str
    ptx_type: str  # "u64", "u32", "s32", "f32", etc.
    is_struct: bool = False   # True for .param .align N .b8 name[M]
    struct_size: int = 0      # byte size for struct params
    struct_align: int = 0     # alignment for struct params

    def cpp_type(self) -> str:
        return PTX_TYPE_TO_CPP.get(self.ptx_type, "uint64_t")


@dataclass
class Instruction:
    """A parsed PTX instruction."""
    predicate: str | None  # e.g. "%p1" or None
    pred_negate: bool      # @!%p1
    opcode: str            # e.g. "add"
    modifiers: list[str]   # e.g. ["lo", "s32"] from add.lo.s32
    operands: list[str]    # raw operand strings
    source_line: int = 0   # line number in original PTX


@dataclass
class Diagnostic:
    """A warning or error from translation."""
    line_number: int
    severity: str   # "warning" or "error"
    message: str

    def __str__(self):
        return f"ptx2cpp: {self.severity}: line {self.line_number}: {self.message}"


@dataclass
class SharedDecl:
    """A .shared memory declaration."""
    name: str
    size: int       # total bytes
    offset: int     # offset within shared memory buffer


@dataclass
class ConstDecl:
    """A .const / __constant__ variable declaration."""
    name: str
    size: int       # total bytes
    align: int      # alignment


@dataclass
class KernelDef:
    """A parsed PTX kernel (.entry)."""
    name: str
    params: list[Param]
    registers: dict[str, int]  # prefix -> count, e.g. {"r": 6, "rd": 11}
    instructions: list[Instruction]
    labels: set[str]
    shared_decls: list[SharedDecl] = field(default_factory=list)
    uses_shared_memory: bool = False
    has_dynamic_shared: bool = False
    uses_warp: bool = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PTX_TYPE_TO_CPP = {
    "u8":  "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "s8":  "int8_t",
    "s16": "int16_t",
    "s32": "int32_t",
    "s64": "int64_t",
    "b8":  "uint8_t",
    "b16": "uint16_t",
    "b32": "uint32_t",
    "b64": "uint64_t",
    "f32": "float",
    "f64": "double",
    "pred": "bool",
}

# Maps PTX register prefix to C++ type
REG_PREFIX_TO_TYPE = {
    "r":  "uint32_t",
    "rd": "uint64_t",
    "f":  "float",
    "fd": "double",
    "p":  "bool",
}

# Special registers
SPECIAL_REGS = {
    "%tid.x": "tid_x", "%tid.y": "tid_y", "%tid.z": "tid_z",
    "%ctaid.x": "ctaid_x", "%ctaid.y": "ctaid_y", "%ctaid.z": "ctaid_z",
    "%ntid.x": "ntid_x", "%ntid.y": "ntid_y", "%ntid.z": "ntid_z",
    "%nctaid.x": "nctaid_x", "%nctaid.y": "nctaid_y", "%nctaid.z": "nctaid_z",
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PTXParser:
    """Parses PTX text into KernelDef structures."""

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0
        self.const_decls: list[ConstDecl] = []

    def parse(self) -> list[KernelDef]:
        kernels = []
        while self.pos < len(self.lines):
            line = self._current_line()
            if ".entry" in line:
                kernels.append(self._parse_kernel())
            else:
                # Parse .const declarations (module-level)
                const_match = re.match(
                    r'\.(?:visible\s+)?\.const\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)\[(\d+)\]',
                    line)
                if const_match:
                    align = int(const_match.group(1)) if const_match.group(1) else 4
                    elem_type = const_match.group(2)
                    cname = const_match.group(3)
                    count = int(const_match.group(4))
                    type_sizes = {"b8": 1, "b16": 2, "b32": 4, "b64": 8,
                                  "u8": 1, "u16": 2, "u32": 4, "u64": 8,
                                  "s8": 1, "s16": 2, "s32": 4, "s64": 8,
                                  "f32": 4, "f64": 8}
                    elem_size = type_sizes.get(elem_type, 1)
                    self.const_decls.append(ConstDecl(cname, elem_size * count, align))
                self.pos += 1
        return kernels

    def _current_line(self) -> str:
        return self.lines[self.pos].strip() if self.pos < len(self.lines) else ""

    def _parse_kernel(self) -> KernelDef:
        # Parse .entry line to get kernel name
        line = self._current_line()
        match = re.search(r'\.entry\s+(\w+)', line)
        name = match.group(1) if match else "unknown_kernel"
        self.pos += 1

        # Parse parameters
        params = []
        while self.pos < len(self.lines):
            line = self._current_line()
            if "{" in line:
                self.pos += 1
                break
            # Struct parameter: .param .align N .b8 name[M]
            struct_param_match = re.search(
                r'\.param\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', line)
            if struct_param_match:
                align = int(struct_param_match.group(1))
                pname = struct_param_match.group(2)
                size = int(struct_param_match.group(3))
                params.append(Param(name=pname, ptx_type="b8",
                                    is_struct=True, struct_size=size,
                                    struct_align=align))
                self.pos += 1
                continue
            param_match = re.search(r'\.param\s+\.(\w+)\s+(\w+)', line)
            if param_match:
                params.append(Param(name=param_match.group(2),
                                    ptx_type=param_match.group(1)))
            self.pos += 1

        # Parse body: registers, instructions, labels, shared memory
        registers: dict[str, int] = {}
        instructions: list[Instruction] = []
        labels: set[str] = set()
        shared_decls: list[SharedDecl] = []
        shared_offset = 0

        while self.pos < len(self.lines):
            line = self._current_line()
            self.pos += 1

            if line == "}" or line.startswith("}"):
                break
            if not line or line.startswith("//"):
                continue

            # Register declarations: .reg .f32 %f<4>;
            reg_match = re.match(r'\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>', line)
            if reg_match:
                ptx_type = reg_match.group(1)
                prefix = reg_match.group(2)
                count = int(reg_match.group(3))
                registers[prefix] = count
                continue

            # Extern shared memory: .extern .shared .align 4 .b8 smem[];
            extern_shared_match = re.match(
                r'\.extern\s+\.shared\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+(\w+)\[\]', line)
            if extern_shared_match:
                sname = extern_shared_match.group(2)
                # Dynamic shared memory — offset is at the end of static shared
                # Size is determined at launch time
                shared_decls.append(SharedDecl(sname, 0, shared_offset))
                continue

            # Static shared memory: .shared .align 4 .b8 smem[1024];
            shared_match = re.match(
                r'\.shared\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+(\w+)\[(\d+)\]', line)
            if shared_match:
                elem_type = shared_match.group(1)
                sname = shared_match.group(2)
                count = int(shared_match.group(3))
                # Calculate byte size from type
                type_sizes = {"b8": 1, "b16": 2, "b32": 4, "b64": 8,
                              "u8": 1, "u16": 2, "u32": 4, "u64": 8,
                              "s8": 1, "s16": 2, "s32": 4, "s64": 8,
                              "f32": 4, "f64": 8}
                elem_size = type_sizes.get(elem_type, 1)
                total_bytes = elem_size * count
                # Align offset
                align = max(elem_size, 4)
                shared_offset = (shared_offset + align - 1) & ~(align - 1)
                shared_decls.append(SharedDecl(sname, total_bytes, shared_offset))
                shared_offset += total_bytes
                continue

            # Labels: $L__BB0_2:
            label_match = re.match(r'(\$?\w+):', line)
            if label_match:
                label_name = label_match.group(1)
                labels.add(label_name)
                # Emit label as a pseudo-instruction
                inst = Instruction(
                    predicate=None, pred_negate=False,
                    opcode="__label__", modifiers=[],
                    operands=[label_name],
                    source_line=self.pos
                )
                instructions.append(inst)
                continue

            # Instructions
            inst = self._parse_instruction(line)
            if inst:
                inst.source_line = self.pos
                instructions.append(inst)

        uses_shared = len(shared_decls) > 0 or any(
            "shared" in inst.modifiers for inst in instructions
            if inst.opcode in ("ld", "st")
        )
        has_dynamic = any(sd.size == 0 for sd in shared_decls)
        uses_warp = any(
            inst.opcode in ("shfl", "vote") for inst in instructions
        )
        return KernelDef(name=name, params=params, registers=registers,
                         instructions=instructions, labels=labels,
                         shared_decls=shared_decls, uses_shared_memory=uses_shared,
                         has_dynamic_shared=has_dynamic, uses_warp=uses_warp)

    def _parse_instruction(self, line: str) -> Instruction | None:
        line = line.rstrip(";").strip()
        if not line:
            return None

        # Handle predicate prefix: @%p1 or @!%p1
        predicate = None
        pred_negate = False
        if line.startswith("@"):
            pred_match = re.match(r'@(!?)(%\w+)\s+', line)
            if pred_match:
                pred_negate = pred_match.group(1) == "!"
                predicate = pred_match.group(2)
                line = line[pred_match.end():]

        # Split opcode.modifiers from operands
        parts = line.split(None, 1)
        if not parts:
            return None

        opcode_parts = parts[0].split(".")
        opcode = opcode_parts[0]
        modifiers = opcode_parts[1:]

        operands = []
        if len(parts) > 1:
            operands = self._parse_operands(parts[1])

        return Instruction(predicate=predicate, pred_negate=pred_negate,
                           opcode=opcode, modifiers=modifiers,
                           operands=operands)

    def _parse_operands(self, text: str) -> list[str]:
        """Parse operands handling [addr+offset] brackets and dst|pred pipes."""
        operands = []
        current = ""
        bracket_depth = 0

        for ch in text:
            if ch == "[":
                bracket_depth += 1
                current += ch
            elif ch == "]":
                bracket_depth -= 1
                current += ch
            elif ch == "," and bracket_depth == 0:
                operands.append(current.strip())
                current = ""
            else:
                current += ch

        if current.strip():
            operands.append(current.strip())

        # Expand pipe-separated operands: %r11|%p2 → ["%r11", "%p2"]
        expanded = []
        for op in operands:
            if "|" in op and not op.startswith("["):
                expanded.extend(p.strip() for p in op.split("|"))
            else:
                expanded.append(op)
        return expanded


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class PTXTranslator:
    """Translates parsed PTX instructions into C++ statements."""

    def __init__(self, kernel: KernelDef):
        self.kernel = kernel
        self.diagnostics: list[Diagnostic] = []
        self.param_map: dict[str, str] = {}  # PTX param name -> C++ param name
        for i, p in enumerate(kernel.params):
            self.param_map[p.name] = f"param_{i}"
        # Shared memory: name → offset in shared buffer
        self.shared_map: dict[str, int] = {}
        for sd in kernel.shared_decls:
            self.shared_map[sd.name] = sd.offset

    def translate_all(self) -> list[str]:
        """Translate all instructions to C++ lines."""
        lines = []
        for inst in self.kernel.instructions:
            cpp = self._translate_one(inst)
            if cpp:
                lines.append(cpp)
        return lines

    def _operand_to_cpp(self, op: str) -> str:
        """Convert a PTX operand to C++ expression."""
        # Special registers
        if op in SPECIAL_REGS:
            return SPECIAL_REGS[op]

        # Register: %r3, %rd5, %f1, %p0
        reg_match = re.match(r'^%(\w+?)(\d+)$', op)
        if reg_match:
            prefix = reg_match.group(1)
            index = reg_match.group(2)
            return f"{prefix}[{index}]"

        # Memory operand: [name], [name+offset], [name+%reg], [%reg], [%reg+offset]
        bracket_match = re.match(r'^\[(.+)\]$', op)
        if bracket_match:
            inner = bracket_match.group(1).strip()
            # Split on '+' to get base and optional offset
            parts = [p.strip() for p in inner.split("+", 1)]
            base_str = parts[0]
            offset_str = parts[1] if len(parts) > 1 else None

            # Resolve base
            if base_str in self.param_map:
                base_cpp = self.param_map[base_str]
            elif base_str in self.shared_map:
                base_cpp = f"((uint64_t)__shared_mem + {self.shared_map[base_str]})"
            elif base_str.startswith("%"):
                base_cpp = self._operand_to_cpp(base_str)
            else:
                base_cpp = base_str

            if offset_str is None:
                return base_cpp

            # Resolve offset (could be number or register)
            if offset_str.startswith("%"):
                offset_cpp = self._operand_to_cpp(offset_str)
            else:
                offset_cpp = offset_str

            return f"({base_cpp} + {offset_cpp})"

        # Immediate integer
        if re.match(r'^-?\d+$', op):
            return op

        # Immediate float
        if re.match(r'^0[fF][0-9a-fA-F]+$', op):
            # PTX float hex literal: 0f3F800000 = 1.0f
            hex_val = int(op[2:], 16)
            import struct
            float_val = struct.unpack('f', struct.pack('I', hex_val))[0]
            return f"{float_val}f"

        # Label
        if op.startswith("$") or op.startswith("L_") or op.startswith("BB"):
            return self._label_to_cpp(op)

        return op

    def _label_to_cpp(self, label: str) -> str:
        """Convert PTX label to valid C++ label."""
        return label.replace("$", "").replace(".", "_")

    def _cast(self, type_mod: str, expr: str) -> str:
        """Wrap expr in a C cast based on PTX type modifier."""
        cpp_type = PTX_TYPE_TO_CPP.get(type_mod)
        if cpp_type:
            return f"({cpp_type})({expr})"
        return expr

    def _get_type_modifier(self) -> str | None:
        """Get the last modifier that looks like a type."""
        return None  # caller provides

    def _ptr_type(self, type_mod: str) -> str:
        """Get pointer type for load/store."""
        cpp_type = PTX_TYPE_TO_CPP.get(type_mod, "uint32_t")
        return cpp_type

    def _translate_one(self, inst: Instruction) -> str | None:
        """Translate a single instruction to C++."""
        op = inst.opcode
        mods = inst.modifiers
        operands = inst.operands

        # Label pseudo-instruction
        if op == "__label__":
            label = self._label_to_cpp(operands[0])
            return f"{label}:;"

        # Get C++ operand expressions
        ops = [self._operand_to_cpp(o) for o in operands]

        # Generate C++ statement
        cpp = self._translate_opcode(op, mods, ops, operands)
        if cpp is None:
            full_inst = f"{op}.{'.'.join(mods)}" if mods else op
            self.diagnostics.append(Diagnostic(
                line_number=inst.source_line,
                severity="warning",
                message=f"unsupported instruction: {full_inst} {', '.join(operands)}"
            ))
            return f"// UNSUPPORTED: {full_inst} {', '.join(operands)}"

        # Wrap in predicate
        if inst.predicate:
            pred_cpp = self._operand_to_cpp(inst.predicate)
            cond = f"!{pred_cpp}" if inst.pred_negate else pred_cpp
            cpp = f"if ({cond}) {{ {cpp} }}"

        return cpp

    def _translate_opcode(self, op: str, mods: list[str],
                          ops: list[str], raw_ops: list[str]) -> str | None:
        """Core translation: opcode + modifiers + operands → C++ statement."""

        # Determine the type modifier (usually the last one)
        type_mod = None
        for m in reversed(mods):
            if m in PTX_TYPE_TO_CPP:
                type_mod = m
                break

        # --- Arithmetic ---
        if op == "add":
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = {self._cast(type_mod, ops[1])} + {self._cast(type_mod, ops[2])};"
            return f"{ops[0]} = {ops[1]} + {ops[2]};"

        if op == "sub":
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = {self._cast(type_mod, ops[1])} - {self._cast(type_mod, ops[2])};"
            return f"{ops[0]} = {ops[1]} - {ops[2]};"

        if op == "mul":
            if "wide" in mods:
                # mul.wide.s32: 32-bit inputs → 64-bit result
                return f"{ops[0]} = (int64_t){self._cast(type_mod, ops[1])} * {self._cast(type_mod, ops[2])};" if type_mod and type_mod.startswith("s") else \
                       f"{ops[0]} = (uint64_t)({ops[1]}) * (uint64_t)({ops[2]});"
            if "lo" in mods:
                return f"{ops[0]} = {self._cast(type_mod, ops[1])} * {self._cast(type_mod, ops[2])};" if type_mod else \
                       f"{ops[0]} = {ops[1]} * {ops[2]};"
            if "hi" in mods and type_mod:
                # mul.hi: return upper half of multiplication
                sign = "int" if type_mod.startswith("s") else "uint"
                wide = "64" if type_mod.endswith("32") else "128"
                shift = int(wide) // 2
                cpp_type = PTX_TYPE_TO_CPP[type_mod]
                return f"{ops[0]} = ({cpp_type})(({sign}{wide}_t)({ops[1]}) * ({sign}{wide}_t)({ops[2]}) >> {shift});"
            return f"{ops[0]} = {ops[1]} * {ops[2]};"

        if op == "mad":
            # mad.lo.s32 d, a, b, c → d = a * b + c
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = {self._cast(type_mod, ops[1])} * {self._cast(type_mod, ops[2])} + {self._cast(type_mod, ops[3])};"
            return f"{ops[0]} = {ops[1]} * {ops[2]} + {ops[3]};"

        if op == "div":
            return f"{ops[0]} = {ops[1]} / {ops[2]};"

        if op == "rem":
            return f"{ops[0]} = {ops[1]} % {ops[2]};"

        if op == "abs":
            if type_mod and type_mod.startswith("f"):
                return f"{ops[0]} = fabsf({ops[1]});" if type_mod == "f32" else f"{ops[0]} = fabs({ops[1]});"
            return f"{ops[0]} = abs({self._cast(type_mod, ops[1])});" if type_mod else f"{ops[0]} = abs({ops[1]});"

        if op == "neg":
            return f"{ops[0]} = -{ops[1]};"

        if op == "min":
            return f"{ops[0]} = ({ops[1]} < {ops[2]}) ? {ops[1]} : {ops[2]};"

        if op == "max":
            return f"{ops[0]} = ({ops[1]} > {ops[2]}) ? {ops[1]} : {ops[2]};"

        if op == "fma":
            # fma.rn.f32 d, a, b, c → d = a*b + c
            return f"{ops[0]} = std::fma({ops[1]}, {ops[2]}, {ops[3]});"

        # --- Bitwise ---
        if op == "and":
            return f"{ops[0]} = {ops[1]} & {ops[2]};"
        if op == "or":
            return f"{ops[0]} = {ops[1]} | {ops[2]};"
        if op == "xor":
            return f"{ops[0]} = {ops[1]} ^ {ops[2]};"
        if op == "not":
            return f"{ops[0]} = ~{ops[1]};"

        if op == "shl":
            return f"{ops[0]} = {ops[1]} << {ops[2]};"
        if op == "shr":
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = {self._cast(type_mod, ops[1])} >> {ops[2]};"
            return f"{ops[0]} = {ops[1]} >> {ops[2]};"

        # --- Move ---
        if op == "mov":
            # Check if source is a shared memory variable name
            # On GPU, shared addresses are 32-bit offsets; we store the offset
            raw_src = raw_ops[1] if len(raw_ops) > 1 else ""
            if raw_src in self.shared_map:
                return f"{ops[0]} = {self.shared_map[raw_src]}; /* shared addr: {raw_src} */"
            return f"{ops[0]} = {ops[1]};"

        # --- Load / Store ---
        if op == "ld":
            # ld.param.u64, ld.global.f32, ld.shared.f32, etc.
            if "param" in mods:
                return f"{ops[0]} = {ops[1]};"
            ptr_type = self._ptr_type(type_mod) if type_mod else "uint32_t"
            if "shared" in mods:
                # Shared memory: operand is offset within __shared_mem
                addr = ops[1]
                if "__shared_mem" in addr:
                    # Already resolved to __shared_mem + offset
                    return f"{ops[0]} = *({ptr_type}*)({addr});"
                else:
                    # Register holding offset
                    return f"{ops[0]} = *({ptr_type}*)(__shared_mem + (uint32_t)({addr}));"
            # Global memory load
            return f"{ops[0]} = *({ptr_type}*)({ops[1]});"

        if op == "st":
            ptr_type = self._ptr_type(type_mod) if type_mod else "uint32_t"
            if "shared" in mods:
                addr = ops[0]
                if "__shared_mem" in addr:
                    return f"*({ptr_type}*)({addr}) = {ops[1]};"
                else:
                    return f"*({ptr_type}*)(__shared_mem + (uint32_t)({addr})) = {ops[1]};"
            # Global memory store
            return f"*({ptr_type}*)({ops[0]}) = {ops[1]};"

        # --- Address conversion ---
        if op == "cvta":
            # cvta.to.global.u64 — in CPU simulation, addresses are the same
            return f"{ops[0]} = {ops[1]};"

        # --- Type conversion ---
        if op == "cvt":
            # cvt.[rounding].dst_type.src_type %dst, %src
            # rounding: rn (nearest), rz (zero), rm (minus inf), rp (plus inf)
            #           rni/rzi/rmi/rpi (integer rounding variants)
            dst_type = None
            src_type = None
            rounding = None
            for m in mods:
                if m in ("rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi"):
                    rounding = m
                elif m in PTX_TYPE_TO_CPP:
                    if dst_type is None:
                        dst_type = m
                    else:
                        src_type = m
            cpp_type = PTX_TYPE_TO_CPP.get(dst_type, "uint32_t") if dst_type else None

            # Float→int with rounding
            if rounding and cpp_type and dst_type and dst_type.startswith(("s", "u")):
                if rounding in ("rni", "rn"):
                    return f"{ops[0]} = ({cpp_type})lrintf({ops[1]});"
                if rounding in ("rzi", "rz"):
                    return f"{ops[0]} = ({cpp_type})({ops[1]});"  # truncation (default C cast)
                if rounding in ("rmi", "rm"):
                    return f"{ops[0]} = ({cpp_type})floorf({ops[1]});"
                if rounding in ("rpi", "rp"):
                    return f"{ops[0]} = ({cpp_type})ceilf({ops[1]});"

            # Float→float with rounding (e.g. f64→f32)
            if rounding and cpp_type and dst_type and dst_type.startswith("f"):
                return f"{ops[0]} = ({cpp_type})({ops[1]});"

            if cpp_type:
                return f"{ops[0]} = ({cpp_type})({ops[1]});"
            return f"{ops[0]} = {ops[1]};"

        # --- Comparison ---
        if op == "setp":
            # setp.ge.s32 %p1, %r1, %r2
            cmp_op = mods[0] if mods else "eq"
            cmp_map = {
                "eq": "==", "ne": "!=",
                "lt": "<",  "le": "<=",
                "gt": ">",  "ge": ">=",
                "lo": "<",  "ls": "<=",
                "hi": ">",  "hs": ">=",
            }
            cmp_sym = cmp_map.get(cmp_op, "==")
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = ({self._cast(type_mod, ops[1])} {cmp_sym} {self._cast(type_mod, ops[2])});"
            return f"{ops[0]} = ({ops[1]} {cmp_sym} {ops[2]});"

        if op == "selp":
            # selp.type %d, %a, %b, %p → d = p ? a : b
            return f"{ops[0]} = {ops[3]} ? {ops[1]} : {ops[2]};"

        # --- Control flow ---
        if op == "bra":
            label = self._label_to_cpp(raw_ops[0])
            return f"goto {label};"

        if op == "ret":
            return "return;"

        # --- Math ---
        if op == "sin":
            return f"{ops[0]} = sinf({ops[1]});"
        if op == "cos":
            return f"{ops[0]} = cosf({ops[1]});"
        if op == "sqrt":
            if type_mod == "f64":
                return f"{ops[0]} = sqrt({ops[1]});"
            return f"{ops[0]} = sqrtf({ops[1]});"
        if op == "rsqrt":
            if type_mod == "f64":
                return f"{ops[0]} = 1.0 / sqrt({ops[1]});"
            return f"{ops[0]} = 1.0f / sqrtf({ops[1]});"
        if op == "lg2":
            return f"{ops[0]} = log2f({ops[1]});"
        if op == "ex2":
            return f"{ops[0]} = exp2f({ops[1]});"
        if op == "rcp":
            if type_mod == "f64":
                return f"{ops[0]} = 1.0 / {ops[1]};"
            return f"{ops[0]} = 1.0f / {ops[1]};"

        # --- Atomic ---
        if op == "atom":
            # atom.global.add.s32 %r1, [%rd2], %r3
            atom_op = None
            for m in mods:
                if m in ("add", "min", "max", "inc", "dec", "cas", "exch",
                         "and", "or", "xor"):
                    atom_op = m
                    break
            ptr_type = self._ptr_type(type_mod) if type_mod else "int32_t"
            if atom_op == "add":
                return f"{ops[0]} = cuda_sim::atomic_add(({ptr_type}*)({ops[1]}), ({ptr_type}){ops[2]});"
            if atom_op == "cas":
                return f"{ops[0]} = cuda_sim::atomic_cas(({ptr_type}*)({ops[1]}), ({ptr_type}){ops[2]}, ({ptr_type}){ops[3]});"
            if atom_op == "exch":
                return f"{ops[0]} = cuda_sim::atomic_exch(({ptr_type}*)({ops[1]}), ({ptr_type}){ops[2]});"
            if atom_op == "min":
                return f"{ops[0]} = cuda_sim::atomic_min(({ptr_type}*)({ops[1]}), ({ptr_type}){ops[2]});"
            if atom_op == "max":
                return f"{ops[0]} = cuda_sim::atomic_max(({ptr_type}*)({ops[1]}), ({ptr_type}){ops[2]});"
            return None

        # --- Barrier ---
        if op == "bar":
            if self.kernel.uses_shared_memory or self.kernel.uses_warp:
                return "__barrier->arrive_and_wait(); /* __syncthreads() */"
            return "/* __syncthreads() — no-op in sequential mode */;"

        # --- Warp primitives ---
        if op == "shfl":
            # After pipe expansion: ops = [dst, pred, src, offset, clamp, mask]
            # or without pred:      ops = [dst, src, offset, clamp, mask]
            variant = None
            for m in mods:
                if m in ("down", "up", "idx", "bfly"):
                    variant = m
                    break
            # Determine dst and src positions
            # If second operand is a predicate (p[N]), it's the pipe-expanded pred
            dst = ops[0]
            if len(ops) >= 6 and "p[" in ops[1]:
                pred = ops[1]
                src = ops[2]
                offset = ops[3]
            else:
                pred = None
                src = ops[1]
                offset = ops[2]

            shfl_call = {
                "down": f"__warp_ctx->shfl_down(__lane_id, {src}, {offset})",
                "up":   f"__warp_ctx->shfl_up(__lane_id, {src}, {offset})",
                "idx":  f"__warp_ctx->shfl_idx(__lane_id, {src}, {offset})",
                "bfly": f"__warp_ctx->shfl_xor(__lane_id, {src}, {offset})",
            }.get(variant)
            if shfl_call is None:
                return None
            result = f"{dst} = {shfl_call};"
            if pred:
                result += f" {pred} = true;"
            return result

        if op == "vote":
            # vote.sync.ballot.b32 %dst, %pred
            if "ballot" in mods:
                return f"{ops[0]} = __warp_ctx->ballot(__lane_id, {ops[1]});"
            if "any" in mods:
                return f"{ops[0]} = __warp_ctx->any(__lane_id, {ops[1]});"
            if "all" in mods:
                return f"{ops[0]} = __warp_ctx->all(__lane_id, {ops[1]});"
            return None

        if op == "match":
            # match.any.sync.b32 %dst, %src, %mask
            # match.all.sync.b32 %dst|%pred, %src, %mask
            if "any" in mods:
                return f"{ops[0]} = __warp_ctx->match_any(__lane_id, {ops[1]});"
            if "all" in mods:
                # ops may have pipe-expanded pred
                if len(ops) >= 3 and "p[" in ops[1]:
                    dst, pred, src = ops[0], ops[1], ops[2]
                    return f"{{ bool __mp; {dst} = __warp_ctx->match_all(__lane_id, {src}, __mp); {pred} = __mp; }}"
                return f"{{ bool __mp; {ops[0]} = __warp_ctx->match_all(__lane_id, {ops[1]}, __mp); }}"
            return None

        if op == "activemask":
            return f"{ops[0]} = cuda_sim::WarpContext::activemask();"

        # --- Bit operations ---
        if op == "popc":
            return f"{ops[0]} = cuda_sim::device_popc({ops[1]});"

        if op == "clz":
            return f"{ops[0]} = cuda_sim::device_clz({ops[1]});"

        if op == "bfind":
            return f"{ops[0]} = cuda_sim::device_bfind({ops[1]});"

        if op == "brev":
            return f"{ops[0]} = cuda_sim::device_brev({ops[1]});"

        if op == "bfe":
            # bfe.u32 %dst, %src, %start, %len
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = cuda_sim::device_bfe_signed((int32_t){ops[1]}, {ops[2]}, {ops[3]});"
            return f"{ops[0]} = cuda_sim::device_bfe({ops[1]}, {ops[2]}, {ops[3]});"

        if op == "bfi":
            # bfi.b32 %dst, %src, %base, %start, %len
            return f"{ops[0]} = cuda_sim::device_bfi({ops[1]}, {ops[2]}, {ops[3]}, {ops[4]});"

        if op == "ffs":
            return f"{ops[0]} = cuda_sim::device_ffs({ops[1]});"

        # --- Additional instructions ---

        # copysign: copy sign of b to magnitude of a
        if op == "copysign":
            if type_mod == "f64":
                return f"{ops[0]} = copysign({ops[1]}, {ops[2]});"
            return f"{ops[0]} = copysignf({ops[1]}, {ops[2]});"

        # slct: select a or b based on sign of c
        if op == "slct":
            # slct.type.cmp_type %dst, %a, %b, %c → dst = (c >= 0) ? a : b
            return f"{ops[0]} = ({ops[3]} >= 0) ? {ops[1]} : {ops[2]};"

        # sad: sum of absolute differences
        if op == "sad":
            # sad.type %dst, %a, %b, %c → dst = |a - b| + c
            if type_mod and type_mod.startswith("s"):
                return f"{ops[0]} = abs((int32_t){ops[1]} - (int32_t){ops[2]}) + {ops[3]};"
            return f"{ops[0]} = (({ops[1]} > {ops[2]}) ? ({ops[1]} - {ops[2]}) : ({ops[2]} - {ops[1]})) + {ops[3]};"

        # prmt: permute bytes from two 32-bit values
        if op == "prmt":
            # prmt.b32 %dst, %a, %b, %selector
            return f"""{{ uint32_t __ab[8]; uint32_t __a = {ops[1]}, __b = {ops[2]}, __s = {ops[3]};
    for (int __i = 0; __i < 4; __i++) __ab[__i] = (__a >> (__i*8)) & 0xFF;
    for (int __i = 0; __i < 4; __i++) __ab[4+__i] = (__b >> (__i*8)) & 0xFF;
    {ops[0]} = 0;
    for (int __i = 0; __i < 4; __i++) {ops[0]} |= __ab[(__s >> (__i*4)) & 0x7] << (__i*8); }}"""

        # testp: test floating point property
        if op == "testp":
            # testp.{finite,infinite,number,notanumber,normal,subnormal} %pred, %src
            prop = mods[0] if mods else ""
            if prop == "finite":
                return f"{ops[0]} = std::isfinite({ops[1]});"
            if prop in ("infinite", "inf"):
                return f"{ops[0]} = std::isinf({ops[1]});"
            if prop == "number":
                return f"{ops[0]} = !std::isnan({ops[1]});"
            if prop in ("notanumber", "nan"):
                return f"{ops[0]} = std::isnan({ops[1]});"
            if prop == "normal":
                return f"{ops[0]} = std::isnormal({ops[1]});"
            if prop == "subnormal":
                return f"{ops[0]} = (std::fpclassify({ops[1]}) == FP_SUBNORMAL);"
            return None

        # red: reduction on global/shared memory (similar to atom but no return)
        if op == "red":
            # red.global.add.s32 [addr], val
            atom_op = None
            for m in mods:
                if m in ("add", "min", "max", "and", "or", "xor", "inc", "dec"):
                    atom_op = m
                    break
            ptr_type = self._ptr_type(type_mod) if type_mod else "int32_t"
            if atom_op == "add":
                return f"cuda_sim::atomic_add(({ptr_type}*)({ops[0]}), ({ptr_type}){ops[1]});"
            if atom_op == "min":
                return f"cuda_sim::atomic_min(({ptr_type}*)({ops[0]}), ({ptr_type}){ops[1]});"
            if atom_op == "max":
                return f"cuda_sim::atomic_max(({ptr_type}*)({ops[0]}), ({ptr_type}){ops[1]});"
            return None

        # exit: terminate thread
        if op == "exit":
            return "return;"

        # trap: trigger error
        if op == "trap":
            return 'fprintf(stderr, "cuda_sim: trap instruction hit\\n"); return;'

        # brkpt: breakpoint (debug)
        if op == "brkpt":
            return "/* breakpoint */;"

        # prefetch / prefetchu: cache hint — no-op on CPU
        if op in ("prefetch", "prefetchu"):
            return "/* prefetch — no-op on CPU */;"

        # isspacep: test address space — always true for global on CPU
        if op == "isspacep":
            return f"{ops[0]} = true; /* address space check — trivially true on CPU */;"

        # --- No-op ---
        if op in ("nop", "membar", "fence"):
            return "/* memory fence — no-op in sequential mode */;"

        return None


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

def generate_cpp(kernels: list[KernelDef],
                  const_decls: list[ConstDecl] | None = None) -> tuple[str, list[Diagnostic]]:
    """Generate complete C++ file from translated kernels.
    Returns (cpp_code, diagnostics)."""
    all_diagnostics: list[Diagnostic] = []
    lines = []
    lines.append("// Auto-generated by ptx2cpp.py — do not edit")
    lines.append("#include <cstdint>")
    lines.append("#include <cmath>")
    lines.append("#include <cstring>")
    lines.append('#include "cuda_sim/runtime.h"')
    lines.append('#include "cuda_sim/device_atomic.h"')

    # Check if any kernel needs threading (shared memory or warp)
    any_threaded = any(k.uses_shared_memory or k.uses_warp for k in kernels)
    if any_threaded:
        lines.append("#include <thread>")
        lines.append("#include <vector>")
        lines.append('#include "cuda_sim/barrier.h"')
    # Always include warp.h — it also has bit ops (popc, clz, bfe, etc.)
    lines.append('#include "cuda_sim/warp.h"')
    lines.append("")

    for kernel in kernels:
        translator = PTXTranslator(kernel)
        cpp_lines = translator.translate_all()
        all_diagnostics.extend(translator.diagnostics)
        num_instructions = len([i for i in kernel.instructions if i.opcode != "__label__"])

        uses_shared = kernel.uses_shared_memory
        uses_warp = kernel.uses_warp
        needs_threading = uses_shared or uses_warp

        # --- Thread function ---
        param_decls = []
        for i, p in enumerate(kernel.params):
            param_decls.append(f"{p.cpp_type()} param_{i}")

        # Thread index parameters
        param_decls.extend([
            "uint32_t tid_x", "uint32_t tid_y", "uint32_t tid_z",
            "uint32_t ctaid_x", "uint32_t ctaid_y", "uint32_t ctaid_z",
            "uint32_t ntid_x", "uint32_t ntid_y", "uint32_t ntid_z",
            "uint32_t nctaid_x", "uint32_t nctaid_y", "uint32_t nctaid_z",
        ])

        # Shared memory parameters
        if uses_shared:
            param_decls.append("uint8_t* __shared_mem")
        if needs_threading:
            param_decls.append("cuda_sim::SimpleBarrier* __barrier")
        if uses_warp:
            param_decls.append("cuda_sim::WarpContext* __warp_ctx")
            param_decls.append("uint32_t __lane_id")

        lines.append(f"static void {kernel.name}_thread(")
        lines.append(f"    {(',{0}'.format(chr(10)) + '    ').join(param_decls)})")
        lines.append("{")

        # Register declarations
        for prefix, count in sorted(kernel.registers.items()):
            cpp_type = REG_PREFIX_TO_TYPE.get(prefix, "uint32_t")
            lines.append(f"    {cpp_type} {prefix}[{count}] = {{}};")
        if kernel.registers:
            lines.append("")

        # Instructions
        for cpp_line in cpp_lines:
            if cpp_line.endswith(":;"):
                lines.append(cpp_line)
            else:
                lines.append(f"    {cpp_line}")

        lines.append("}")
        lines.append("")

        # --- Launch wrapper ---
        launch_params = []
        for i, p in enumerate(kernel.params):
            launch_params.append(f"{p.cpp_type()} param_{i}")
        launch_params.append("cuda_sim::dim3 grid")
        launch_params.append("cuda_sim::dim3 block")
        if kernel.has_dynamic_shared:
            launch_params.append("size_t shared_mem_bytes = 0")

        lines.append(f'extern "C"')
        lines.append(f"void {kernel.name}_launch(")
        lines.append(f"    {(',{0}'.format(chr(10)) + '    ').join(launch_params)})")
        lines.append("{")

        # Calculate shared memory size (static part)
        static_shared_size = 0
        if uses_shared:
            for sd in kernel.shared_decls:
                end = sd.offset + sd.size
                if end > static_shared_size:
                    static_shared_size = end
            if static_shared_size == 0 and not kernel.has_dynamic_shared:
                static_shared_size = 1024

        call_args = [f"param_{i}" for i in range(len(kernel.params))]
        call_args.extend([
            "tx", "ty", "tz",
            "bx", "by", "bz",
            "block.x", "block.y", "block.z",
            "grid.x", "grid.y", "grid.z",
        ])

        if needs_threading:
            # Multi-threaded: spawn real threads per block
            if uses_shared:
                call_args.append("shared_mem")
            call_args.append("&barrier")
            if uses_warp:
                call_args.append("&warp_ctxs[thread_idx / 32]")
                call_args.append("thread_idx % 32")

            lines.append("    for (uint32_t bz = 0; bz < grid.z; ++bz)")
            lines.append("    for (uint32_t by = 0; by < grid.y; ++by)")
            lines.append("    for (uint32_t bx = 0; bx < grid.x; ++bx) {")
            if uses_shared:
                if kernel.has_dynamic_shared:
                    lines.append(f"        size_t __total_shared = {static_shared_size} + shared_mem_bytes;")
                    lines.append("        std::vector<uint8_t> shared_vec(__total_shared, 0);")
                    lines.append("        uint8_t* shared_mem = shared_vec.data();")
                else:
                    lines.append(f"        uint8_t shared_mem[{static_shared_size}] = {{}};")
            lines.append("        uint32_t num_threads = block.x * block.y * block.z;")
            lines.append("        cuda_sim::SimpleBarrier barrier(num_threads);")
            if uses_warp:
                lines.append("        uint32_t num_warps = (num_threads + 31) / 32;")
                lines.append("        std::vector<cuda_sim::WarpContext> warp_ctxs(num_warps);")
                lines.append("        // Resize warp barriers to actual warp size")
                lines.append("        for (uint32_t w = 0; w < num_warps; w++) {")
                lines.append("            uint32_t warp_size = (w == num_warps - 1 && num_threads % 32 != 0)")
                lines.append("                ? num_threads % 32 : 32;")
                lines.append("            warp_ctxs[w].barrier.reset(warp_size);")
                lines.append("        }")
            lines.append("        std::vector<std::thread> threads;")
            lines.append("        threads.reserve(num_threads);")
            lines.append("        uint32_t thread_idx = 0;")
            lines.append("        for (uint32_t tz = 0; tz < block.z; ++tz)")
            lines.append("        for (uint32_t ty = 0; ty < block.y; ++ty)")
            lines.append("        for (uint32_t tx = 0; tx < block.x; ++tx) {")
            lines.append(f"            threads.emplace_back({kernel.name}_thread, {', '.join(call_args)});")
            lines.append("            thread_idx++;")
            lines.append("        }")
            lines.append("        for (auto& t : threads) t.join();")
            lines.append("    }")
        else:
            # Sequential: simple nested loop (fast, no threading overhead)
            lines.append("    for (uint32_t bz = 0; bz < grid.z; ++bz)")
            lines.append("    for (uint32_t by = 0; by < grid.y; ++by)")
            lines.append("    for (uint32_t bx = 0; bx < grid.x; ++bx)")
            lines.append("        for (uint32_t tz = 0; tz < block.z; ++tz)")
            lines.append("        for (uint32_t ty = 0; ty < block.y; ++ty)")
            lines.append("        for (uint32_t tx = 0; tx < block.x; ++tx)")
            lines.append(f"            {kernel.name}_thread({', '.join(call_args)});")

        lines.append("}")
        lines.append("")

        # --- Convenience wrapper: void* for pointer params ---
        # u64 params are likely pointers; generate a void* overload
        has_u64 = any(p.ptx_type in ("u64",) for p in kernel.params)
        if has_u64:
            wrapper_params = []
            cast_args = []
            for i, p in enumerate(kernel.params):
                if p.ptx_type in ("u64",):
                    wrapper_params.append(f"const void* param_{i}")
                    cast_args.append(f"(uint64_t)param_{i}")
                else:
                    wrapper_params.append(f"{p.cpp_type()} param_{i}")
                    cast_args.append(f"param_{i}")
            wrapper_params.append("cuda_sim::dim3 grid")
            wrapper_params.append("cuda_sim::dim3 block")
            cast_args.extend(["grid", "block"])

            lines.append(f"void {kernel.name}_launch(")
            lines.append(f"    {(',{0}'.format(chr(10)) + '    ').join(wrapper_params)})")
            lines.append("{")
            lines.append(f"    {kernel.name}_launch({', '.join(cast_args)});")
            lines.append("}")
            lines.append("")

        # --- Generic entry for cuLaunchKernel (void** args unpacking) ---
        lines.append(f'extern "C"')
        lines.append(f"void {kernel.name}_launch_generic(void** args,")
        lines.append(f"    uint32_t gx, uint32_t gy, uint32_t gz,")
        lines.append(f"    uint32_t bx, uint32_t by, uint32_t bz,")
        lines.append(f"    uint32_t shared_bytes)")
        lines.append("{")

        # Unpack args by type
        generic_call_args = []
        for i, p in enumerate(kernel.params):
            if p.is_struct:
                # Struct: local array + memcpy from args[i]
                lines.append(f"    uint8_t __p{i}[{p.struct_size}];")
                lines.append(f"    std::memcpy(__p{i}, args[{i}], {p.struct_size});")
                # The typed launch expects the struct passed as uint8_t* casted to
                # the param type (which for struct params is just a u64 holding the pointer).
                # However, since struct params in PTX are loaded via ld.param into registers,
                # and our _thread function takes them as uint64_t (address), we pass the address.
                generic_call_args.append(f"(uint64_t)(uintptr_t)__p{i}")
            else:
                cpp_t = p.cpp_type()
                lines.append(f"    {cpp_t} __p{i} = *({cpp_t}*)args[{i}];")
                generic_call_args.append(f"__p{i}")

        generic_call_args.append("cuda_sim::dim3{gx, gy, gz}")
        generic_call_args.append("cuda_sim::dim3{bx, by, bz}")
        if kernel.has_dynamic_shared:
            generic_call_args.append("(size_t)shared_bytes")

        lines.append(f"    {kernel.name}_launch({', '.join(generic_call_args)});")
        lines.append("}")
        lines.append("")

    # --- __constant__ variable globals and symbol lookup ---
    if const_decls:
        lines.append("// --- __constant__ variables ---")
        for cd in const_decls:
            lines.append(f"alignas({cd.align}) static uint8_t {cd.name}[{cd.size}];")
        lines.append("")

        lines.append('extern "C" void* __cuda_sim_get_symbol(const char* name) {')
        for cd in const_decls:
            lines.append(f'    if (std::strcmp(name, "{cd.name}") == 0) return {cd.name};')
        lines.append("    return nullptr;")
        lines.append("}")
        lines.append("")

    return "\n".join(lines), all_diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_header(kernels: list[KernelDef]) -> str:
    """Generate a .h header with launch function declarations."""
    lines = []
    lines.append("// Auto-generated by ptx2cpp.py — do not edit")
    lines.append("#pragma once")
    lines.append('#include <cstdint>')
    lines.append('#include <cuda_runtime.h>')
    lines.append("")

    for kernel in kernels:
        dyn_param = ", size_t shared_mem_bytes = 0" if kernel.has_dynamic_shared else ""

        # extern "C" raw launch function
        raw_params = []
        for i, p in enumerate(kernel.params):
            raw_params.append(f"{p.cpp_type()} param_{i}")
        raw_params.append("dim3 grid")
        raw_params.append("dim3 block")
        if kernel.has_dynamic_shared:
            raw_params.append("size_t shared_mem_bytes = 0")

        lines.append(f'extern "C" void {kernel.name}_launch(')
        lines.append(f"    {(',{0}'.format(chr(10)) + '    ').join(raw_params)});")
        lines.append("")

        # void* convenience overload for pointer params
        has_u64 = any(p.ptx_type in ("u64",) for p in kernel.params)
        if has_u64:
            wrapper_params = []
            for i, p in enumerate(kernel.params):
                if p.ptx_type in ("u64",):
                    wrapper_params.append(f"const void* param_{i}")
                else:
                    wrapper_params.append(f"{p.cpp_type()} param_{i}")
            wrapper_params.append("dim3 grid")
            wrapper_params.append("dim3 block")
            if kernel.has_dynamic_shared:
                wrapper_params.append("size_t shared_mem_bytes = 0")

            lines.append(f"void {kernel.name}_launch(")
            lines.append(f"    {(',{0}'.format(chr(10)) + '    ').join(wrapper_params)});")
            lines.append("")

        # Generic entry for cuLaunchKernel
        lines.append(f'extern "C" void {kernel.name}_launch_generic(void** args,')
        lines.append(f"    uint32_t gx, uint32_t gy, uint32_t gz,")
        lines.append(f"    uint32_t bx, uint32_t by, uint32_t bz,")
        lines.append(f"    uint32_t shared_bytes);")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Translate PTX to C++")
    parser.add_argument("input", help="Input .ptx file")
    parser.add_argument("-o", "--output", help="Output .cpp file (default: stdout)")
    parser.add_argument("-H", "--header", help="Output .h header file with launch declarations")
    parser.add_argument("--strict", action="store_true",
                        help="Treat unsupported instructions as errors")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        ptx_text = f.read()

    ptx_parser = PTXParser(ptx_text)
    kernels = ptx_parser.parse()

    if not kernels:
        print("Error: no kernels found in PTX file", file=sys.stderr)
        sys.exit(1)

    cpp_code, diagnostics = generate_cpp(kernels, ptx_parser.const_decls)

    # Print diagnostics
    for d in diagnostics:
        print(str(d), file=sys.stderr)

    # Summary
    num_warnings = sum(1 for d in diagnostics if d.severity == "warning")
    total_insts = sum(
        len([i for i in k.instructions if i.opcode != "__label__"])
        for k in kernels
    )
    summary = f"Generated {args.output or '<stdout>'} ({len(kernels)} kernel(s), {total_insts} instructions"
    if num_warnings:
        summary += f", {num_warnings} warning(s)"
    summary += ")"
    print(summary, file=sys.stderr)

    # Strict mode: fail on warnings
    if args.strict and diagnostics:
        print("Error: unsupported instructions found (--strict mode)", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(cpp_code)
    else:
        print(cpp_code)

    if args.header:
        header_code = generate_header(kernels)
        with open(args.header, "w") as f:
            f.write(header_code)
        print(f"Generated header: {args.header}", file=sys.stderr)


if __name__ == "__main__":
    main()
