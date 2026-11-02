"""TLA 3-in-1 test: vgather per-lane indexed loads with masks on u32.

Lane i of the result reads base[index[i]]; the reference is exact.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask gather, 1 prefix-masked gather (ZEROING).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, pool=256)
SIGNATURE = {"X": "*u32", "I": "*i32", "O": "*u32"}


@tla.jit
def vgather(X, I, O, N: tl.constexpr, ACTIVE: tl.constexpr, POOL: tl.constexpr):
    x = tla.alloc_buffer((POOL, ), tl.uint32, space="ub")
    ib = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((2, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (POOL, )), x)
    tla.copy(tla.view(I, (N, )), ib)
    with tla.scope.simd():
        idx = tla.simd.vlds(ib[0])
        prefix = tla.simd.pge(ACTIVE, N)
        tla.simd.vsts(o[0, 0], tla.simd.vgather(x, idx))
        tla.simd.vsts(o[1, 0], tla.simd.vgather(x, idx, prefix))
    tla.copy(o, tla.view(O, (2, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"], "POOL": SHAPE["pool"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vgather, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vgather") == 2
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vgather", ))
    print("IR PASS: vgather full/prefix-masked")


def _launch():
    s = SHAPE
    x = torch.arange(s["pool"], dtype=torch.int64) * 3 + 7
    idx = (torch.arange(s["lanes"]) * 37 + 11) % s["pool"]
    out = torch.empty((2, s["lanes"]), dtype=torch.uint32, device="npu")
    vgather[(1, )](x.to(torch.uint32).to("npu"), idx.to(torch.int32).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    full = x[idx]
    ref = torch.zeros((2, s["lanes"]), dtype=torch.int64)
    ref[0] = full
    ref[1, :s["active"]] = full[:s["active"]]
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vgather generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("VGATHER OK: per-lane indexed loads exact values")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vgather", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
