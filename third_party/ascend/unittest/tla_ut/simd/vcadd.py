"""TLA 3-in-1 test: vcadd adjacent-pair add (composition: vdintlv + vadd).

Contract: the low N/2 lanes hold src[2i] + src[2i+1]; the upper half is
undefined (same low-lanes-valid contract as vcmax/vcmin), so only the low
half is compared against the torch reference.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask pairs (low half), 1 prefix-masked pairs (low half).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32)
SIGNATURE = {"X": "*fp32", "O": "*fp32"}


@tla.jit
def vcadd(X, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((2, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        v = tla.simd.vlds(x[0])
        prefix = tla.simd.pge(ACTIVE, N)
        tla.simd.vsts(o[0, 0], tla.simd.vcadd(v))
        tla.simd.vsts(o[1, 0], tla.simd.vcadd(v, prefix))
    tla.copy(o, tla.view(O, (2, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vcadd, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vdintlv") == 2, "vcadd composes vdintlv per row"
    assert text.count("ave.hir.vadd") == 2, "vcadd composes vadd per row"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vdintlv", ))
    print("IR PASS: vcadd composed vdintlv+vadd")


def _launch():
    s = SHAPE
    x = torch.randn(s["lanes"], dtype=torch.float32)
    out = torch.empty((2, s["lanes"]), dtype=torch.float32, device="npu")
    vcadd[(1, )](x.to("npu"), out, **constexprs())
    torch.npu.synchronize()
    pairs = (x[0::2] + x[1::2])
    half = s["lanes"] // 2
    ref = torch.zeros((2, s["lanes"]), dtype=torch.float32)
    ref[0, :half] = pairs
    ref[1, :min(s["active"], half)] = pairs[:min(s["active"], half)]
    return out, ref, half


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vcadd generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref, half = _launch()
    got = out.cpu()
    torch.testing.assert_close(got[:, :half], ref[:, :half], rtol=1e-6, atol=1e-6)
    print("VCADD OK: adjacent-pair add, low half exact")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vcadd", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
