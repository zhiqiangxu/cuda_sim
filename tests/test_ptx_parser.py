#!/usr/bin/env python3
"""Unit tests for ptx2cpp.py parser and translator."""

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from ptx2cpp import PTXParser, PTXTranslator, generate_cpp


def test_parse_kernel_name():
    ptx = """
.visible .entry myKernel(
    .param .u32 myKernel_param_0
) {
    .reg .b32 %r<2>;
    ld.param.u32 %r1, [myKernel_param_0];
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    assert len(kernels) == 1
    assert kernels[0].name == "myKernel"
    print("PASS: test_parse_kernel_name")


def test_parse_params():
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0,
    .param .u32 foo_param_1,
    .param .f32 foo_param_2
) {
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    k = kernels[0]
    assert len(k.params) == 3
    assert k.params[0].ptx_type == "u64"
    assert k.params[1].ptx_type == "u32"
    assert k.params[2].ptx_type == "f32"
    print("PASS: test_parse_params")


def test_parse_registers():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<8>;
    .reg .b64  %rd<4>;
    .reg .f32  %f<3>;
    .reg .pred %p<2>;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    k = kernels[0]
    assert k.registers["r"] == 8
    assert k.registers["rd"] == 4
    assert k.registers["f"] == 3
    assert k.registers["p"] == 2
    print("PASS: test_parse_registers")


def test_parse_instructions():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<4>;
    .reg .pred %p<2>;
    mov.u32       %r1, %tid.x;
    add.u32       %r2, %r1, %r3;
    setp.ge.s32   %p1, %r1, %r2;
    @%p1 bra      $DONE;
$DONE:
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    k = kernels[0]
    # mov, add, setp, @%p1 bra, label, ret = 6 instructions
    assert len(k.instructions) == 6, f"Expected 6, got {len(k.instructions)}"
    assert k.instructions[0].opcode == "mov"
    assert k.instructions[1].opcode == "add"
    assert k.instructions[2].opcode == "setp"
    assert k.instructions[3].opcode == "bra"
    assert k.instructions[3].predicate == "%p1"
    assert k.instructions[4].opcode == "__label__"
    assert k.instructions[5].opcode == "ret"
    print("PASS: test_parse_instructions")


def test_parse_predicate_negate():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<2>;
    .reg .pred %p<2>;
    @!%p1 mov.u32 %r1, %r0;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    inst = kernels[0].instructions[0]
    assert inst.pred_negate is True
    assert inst.predicate == "%p1"
    print("PASS: test_parse_predicate_negate")


def test_translate_arithmetic():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<6>;
    .reg .f32  %f<4>;
    add.u32       %r1, %r2, %r3;
    sub.f32       %f1, %f2, %f3;
    mul.lo.s32    %r4, %r1, %r2;
    mad.lo.s32    %r5, %r1, %r2, %r3;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    assert "r[1] = r[2] + r[3];" in lines[0]
    assert "f[1] = f[2] - f[3];" in lines[1]
    assert "r[4]" in lines[2] and "*" in lines[2]
    assert "r[5]" in lines[3] and "*" in lines[3] and "+" in lines[3]
    print("PASS: test_translate_arithmetic")


def test_translate_memory():
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b64  %rd<4>;
    .reg .f32  %f<2>;
    ld.param.u64  %rd1, [foo_param_0];
    ld.global.f32 %f1, [%rd1];
    st.global.f32 [%rd2], %f1;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    assert "rd[1] = param_0;" in lines[0]
    assert "*(float*)" in lines[1]
    assert "*(float*)" in lines[2]
    print("PASS: test_translate_memory")


def test_translate_branch():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<2>;
    .reg .pred %p<2>;
    setp.ge.s32   %p1, %r0, %r1;
    @%p1 bra      $DONE;
    bra            $LOOP;
$DONE:
    ret;
$LOOP:
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    assert "p[1] =" in lines[0] and ">=" in lines[0]
    assert "if (p[1])" in lines[1] and "goto" in lines[1]
    assert "goto LOOP;" in lines[2]
    assert "DONE:;" in lines[3]
    print("PASS: test_translate_branch")


def test_translate_special_regs():
    ptx = """
.visible .entry foo(
    .param .u32 foo_param_0
) {
    .reg .b32  %r<4>;
    mov.u32  %r0, %tid.x;
    mov.u32  %r1, %ctaid.x;
    mov.u32  %r2, %ntid.x;
    mov.u32  %r3, %tid.y;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    assert "tid_x" in lines[0]
    assert "ctaid_x" in lines[1]
    assert "ntid_x" in lines[2]
    assert "tid_y" in lines[3]
    print("PASS: test_translate_special_regs")


def test_translate_cvta():
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b64  %rd<3>;
    ld.param.u64       %rd1, [foo_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    # cvta is a no-op in CPU simulation
    assert "rd[2] = rd[1];" in lines[1]
    print("PASS: test_translate_cvta")


def test_translate_mov_b64():
    """Test mov.b64 split and merge — the bug that caused ProgPow nonce insensitivity."""
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b32  %r<4>;
    .reg .b64  %rd<3>;
    ld.param.u64    %rd0, [foo_param_0];
    mov.b64 {%r0, %r1}, %rd0;
    mov.b64 %rd1, {%r2, %r3};
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    # Split: rd → {lo, hi}
    split_line = lines[1]
    assert "(uint32_t)(rd[0])" in split_line and ">> 32" in split_line, \
        f"mov.b64 split failed: {split_line}"

    # Merge: {lo, hi} → rd
    merge_line = lines[2]
    assert "rd[1]" in merge_line and "<< 32" in merge_line and "r[2]" in merge_line, \
        f"mov.b64 merge failed: {merge_line}"

    print("PASS: test_translate_mov_b64")


def test_translate_shf():
    """Test funnel shift (shf.l/r.wrap.b32) — used heavily in ProgPow keccak."""
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b32  %r<6>;
    shf.l.wrap.b32 %r0, %r1, %r2, %r3;
    shf.r.wrap.b32 %r4, %r1, %r2, %r5;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    # Left shift should use <<
    assert "<<" in lines[0] and ">>" in lines[0], f"shf.l failed: {lines[0]}"
    # Right shift
    assert ">>" in lines[1], f"shf.r failed: {lines[1]}"

    print("PASS: test_translate_shf")


def test_translate_v4_load():
    """Test vector load ld.global.v4.u32 — used for DAG access in ProgPow."""
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b32  %r<5>;
    .reg .b64  %rd<2>;
    ld.param.u64        %rd0, [foo_param_0];
    ld.global.v4.u32    {%r0, %r1, %r2, %r3}, [%rd0];
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    v4_line = lines[1]
    assert "r[0]" in v4_line and "r[1]" in v4_line and "r[2]" in v4_line and "r[3]" in v4_line, \
        f"v4 load failed: {v4_line}"
    assert "[0]" in v4_line and "[1]" in v4_line and "[2]" in v4_line and "[3]" in v4_line, \
        f"v4 load missing indices: {v4_line}"

    print("PASS: test_translate_v4_load")


def test_translate_func():
    """Test .func parsing and call sequence translation."""
    ptx = """
.func (.param .b64 func_retval0) helper(
    .param .b64 helper_param_0,
    .param .b32 helper_param_1
)
{
    .reg .b32  %r<2>;
    .reg .b64  %rd<2>;
    ld.param.u64 %rd0, [helper_param_0];
    ld.param.u32 %r0, [helper_param_1];
    add.u64      %rd1, %rd0, 1;
    st.param.b64 [func_retval0+0], %rd1;
    ret;
}

.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b64  %rd<3>;
    ld.param.u64 %rd0, [foo_param_0];
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64 [param0+0], %rd0;
    .param .b32 param1;
    st.param.b32 [param1+0], 42;
    .param .b64 retval0;
    call.uni (retval0),
    helper,
    (
    param0,
    param1
    );
    ld.param.b64 %rd1, [retval0+0];
    } // callseq 0
    ret;
}
"""
    parser = PTXParser(ptx)
    kernels = parser.parse()

    assert len(parser.func_defs) == 1, "Should parse 1 .func"
    assert parser.func_defs[0].name == "helper"
    assert parser.func_defs[0].ret_type == "b64"

    assert len(kernels) == 1, "Should parse 1 .entry"

    # Check that callseq is translated
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()
    callseq_text = "\n".join(lines)
    assert "helper(" in callseq_text, f"Call to helper not found in: {callseq_text}"
    assert "rd[1]" in callseq_text, f"Return value not captured: {callseq_text}"

    print("PASS: test_translate_func")


def test_translate_atom_inc():
    """Test atom.global.inc — used for output buffer counting in ProgPow."""
    ptx = """
.visible .entry foo(
    .param .u64 foo_param_0
) {
    .reg .b32  %r<2>;
    .reg .b64  %rd<2>;
    ld.param.u64      %rd0, [foo_param_0];
    atom.global.inc.u32 %r0, [%rd0], -1;
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    t = PTXTranslator(kernels[0])
    lines = t.translate_all()

    assert "atomic_inc" in lines[1], f"atom.inc failed: {lines[1]}"
    print("PASS: test_translate_atom_inc")


def test_generate_launch_wrapper():
    ptx = """
.visible .entry myKernel(
    .param .u64 myKernel_param_0,
    .param .u32 myKernel_param_1
) {
    ret;
}
"""
    kernels = PTXParser(ptx).parse()
    cpp, _ = generate_cpp(kernels)

    assert "myKernel_thread" in cpp
    assert "myKernel_launch" in cpp
    assert 'extern "C"' in cpp
    assert "grid.z" in cpp
    assert "block.x" in cpp
    print("PASS: test_generate_launch_wrapper")


def test_end_to_end_compile():
    """Generate C++ from vectorAdd PTX and compile it."""
    ptx_path = os.path.join(os.path.dirname(__file__),
                            "..", "examples", "vector_add", "kernel.ptx")
    if not os.path.exists(ptx_path):
        print("SKIP: test_end_to_end_compile (kernel.ptx not found)")
        return

    with open(ptx_path) as f:
        ptx = f.read()

    kernels = PTXParser(ptx).parse()
    cpp, _ = generate_cpp(kernels)

    # Write to temp file and try to compile
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(cpp)
        tmp_cpp = f.name

    include_dir = os.path.join(os.path.dirname(__file__), "..", "include")
    tmp_out = tmp_cpp.replace(".cpp", ".o")

    try:
        result = subprocess.run(
            ["g++", "-std=c++17", "-c", f"-I{include_dir}", tmp_cpp, "-o", tmp_out],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"FAIL: test_end_to_end_compile\n{result.stderr}")
        else:
            print("PASS: test_end_to_end_compile")
    finally:
        os.unlink(tmp_cpp)
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


def test_end_to_end_run():
    """Build and run the complete vectorAdd example."""
    example_dir = os.path.join(os.path.dirname(__file__),
                               "..", "examples", "vector_add")
    include_dir = os.path.join(os.path.dirname(__file__), "..", "include")

    main_cpp = os.path.join(example_dir, "main.cpp")
    kernel_cpp = os.path.join(example_dir, "kernel_cpu.cpp")

    if not os.path.exists(kernel_cpp):
        print("SKIP: test_end_to_end_run (kernel_cpu.cpp not found)")
        return

    with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
        tmp_bin = f.name

    try:
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{include_dir}",
             main_cpp, kernel_cpp, "-o", tmp_bin],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"FAIL: test_end_to_end_run (compile)\n{result.stderr}")
            return

        result = subprocess.run([tmp_bin], capture_output=True, text=True)
        if result.returncode != 0 or "PASS" not in result.stdout:
            print(f"FAIL: test_end_to_end_run (run)\n{result.stdout}\n{result.stderr}")
        else:
            print(f"PASS: test_end_to_end_run → {result.stdout.strip()}")
    finally:
        if os.path.exists(tmp_bin):
            os.unlink(tmp_bin)


if __name__ == "__main__":
    test_parse_kernel_name()
    test_parse_params()
    test_parse_registers()
    test_parse_instructions()
    test_parse_predicate_negate()
    test_translate_arithmetic()
    test_translate_memory()
    test_translate_branch()
    test_translate_special_regs()
    test_translate_cvta()
    test_translate_mov_b64()
    test_translate_shf()
    test_translate_v4_load()
    test_translate_func()
    test_translate_atom_inc()
    test_generate_launch_wrapper()
    test_end_to_end_compile()
    test_end_to_end_run()
    print("\nAll tests done.")
