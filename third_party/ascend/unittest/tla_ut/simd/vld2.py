"""TLA 3-in-1 test: vld2 de-interleaving two-vector load on f32.

vld2(x[0], dist=DINTLV_B32) returns (even, odd) halves of one strided
access.  Reference hypothesis: the memory is de-interleaved, i.e.
even[i] = x[2i] and odd[i] = x[2i+1].

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 even lanes, 1 odd lanes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, total=128)
SIGNATURE = {"X": "*fp32", "O": "*fp32"}


@tla.jit
def vld2(X, O, N: tl.constexpr, TOTAL: tl.constexpr):
    x = tla.alloc_buffer((TOTAL, ), tl.float32, space="ub")
    o = tla.alloc_buffer((2, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (TOTAL, )), x)
    with tla.scope.simd():
        even, odd = tla.simd.vld2(x[0])
        tla.simd.vsts(o[0, 0], even)
        tla.simd.vsts(o[1, 0], odd)
    tla.copy(o, tla.view(O, (2, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "TOTAL": SHAPE["total"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vld2, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vload") == 1
    assert "dintlv" in text.lower(), "vld2 must use a DINTLV load distribution"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vload", ))
    print("IR PASS: vld2 DINTLV two-vector load")


def _launch():
    s = SHAPE
    x = torch.arange(s["total"], dtype=torch.float32)
    out = torch.empty((2, s["lanes"]), dtype=torch.float32, device="npu")
    vld2[(1, )](x.to("npu"), out, **constexprs())
    torch.npu.synchronize()
    ref = torch.zeros((2, s["lanes"]), dtype=torch.float32)
    ref[0] = x[0::2]
    ref[1] = x[1::2]
    return out, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vld2 generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    print("VLD2 OK: de-interleaved even/odd halves exact values")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vld2", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
