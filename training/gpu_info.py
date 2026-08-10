#!/usr/bin/env python3
"""
Standalone GPU capability probe.

Answers one question: "what can THIS GPU actually do?" — with measurements,
not spec sheets. Designed to be dropped onto any machine (local box, RunPod,
Vast.ai, ...) with only PyTorch installed. It knows nothing about this
project's configs or datasets — that is doctor.py's job.

Everything is probed EMPIRICALLY (kernels are actually executed), because
compute-capability numbers lie across vendors: ROCm reports cc 12.0 on RDNA4,
which is not Blackwell.

Usage:
    python gpu_info.py            # full probe (~1-2 min)
    python gpu_info.py --quick    # identity + feature detection only (~10 s)
    python gpu_info.py --burn 30  # longer sustained-load test (default 10 s)
"""

import argparse
import contextlib
import os
import platform
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SEP = '=' * 64
IS_ROCM = torch.version.hip is not None
GB = 1024**3

# Red flags collected across sections, repeated in the final verdict.
FLAGS: list[str] = []
# Measurements stashed for the final verdict.
VERDICT: dict = {}


def flag(msg):
    FLAGS.append(msg)
    print(f'  🚩 {msg}')


def header(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)


def _sh(cmd, timeout=15):
    """Run a shell command, return stdout ('' on any failure)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ''


def _event_timer(fn, iters=10):
    """Average elapsed ms over `iters` calls, using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ============================================================================
# SMI helpers (nvidia-smi on CUDA, rocm-smi on ROCm) — all best-effort
# ============================================================================


def smi_gpu_info(index=0) -> dict:
    """Driver/power/thermal/PCIe info for one GPU. Empty dict if unavailable."""
    info = {}
    if not IS_ROCM and shutil.which('nvidia-smi'):
        fields = (
            'driver_version,power.limit,power.draw,temperature.gpu,'
            'clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,'
            'pcie.link.gen.max,pcie.link.width.max,ecc.mode.current'
        )
        out = _sh(f'nvidia-smi --query-gpu={fields} --format=csv,noheader -i {index}')
        if out:
            vals = [v.strip() for v in out.split(',')]
            keys = fields.split(',')
            info = dict(zip(keys, vals, strict=False))
            info['pcie'] = f'Gen{info.get("pcie.link.gen.current", "?")} x{info.get("pcie.link.width.current", "?")}'
            info['pcie_max'] = f'Gen{info.get("pcie.link.gen.max", "?")} x{info.get("pcie.link.width.max", "?")}'
    elif IS_ROCM and shutil.which('rocm-smi'):
        try:
            import json as _json

            out = _sh('rocm-smi --showtemp --showpower --showmaxpower --showdriverversion --showbus --json')
            data = _json.loads(out) if out else {}
            card = data.get(f'card{index}', {})
            info = {
                'driver_version': data.get('system', {}).get('Driver version', ''),
                'temperature.gpu': card.get('Temperature (Sensor junction) (C)', card.get('Temperature (Sensor edge) (C)', '')),
                'power.draw': card.get('Average Graphics Package Power (W)', ''),
                'power.limit': card.get('Max Graphics Package Power (W)', ''),
            }
            # PCIe link via sysfs (rocm-smi gives us the bus address)
            bus = card.get('PCI Bus', '')
            if bus:
                base = f'/sys/bus/pci/devices/{bus.lower()}'
                try:
                    with open(f'{base}/current_link_speed') as f:
                        speed = f.read().strip()
                    with open(f'{base}/current_link_width') as f:
                        width = f.read().strip()
                    info['pcie'] = f'{speed} x{width}'
                    with open(f'{base}/max_link_speed') as f:
                        mspeed = f.read().strip()
                    with open(f'{base}/max_link_width') as f:
                        mwidth = f.read().strip()
                    info['pcie_max'] = f'{mspeed} x{mwidth}'
                except OSError:
                    pass
        except Exception:
            pass
    return {k: v for k, v in info.items() if v not in ('', 'N/A', '[N/A]')}


NVIDIA_ARCH = {
    (5, 0): 'Maxwell',
    (5, 2): 'Maxwell',
    (6, 0): 'Pascal',
    (6, 1): 'Pascal',
    (7, 0): 'Volta',
    (7, 5): 'Turing',
    (8, 0): 'Ampere (A100)',
    (8, 6): 'Ampere (RTX 30)',
    (8, 7): 'Ampere (Orin)',
    (8, 9): 'Ada Lovelace (RTX 40)',
    (9, 0): 'Hopper (H100)',
    (10, 0): 'Blackwell (B100/B200)',
    (12, 0): 'Blackwell (RTX 50)',
}

AMD_ARCH = {
    'gfx906': 'Vega 20 / MI50 (GCN)',
    'gfx908': 'CDNA 1 (MI100)',
    'gfx90a': 'CDNA 2 (MI200)',
    'gfx940': 'CDNA 3 (MI300)',
    'gfx941': 'CDNA 3 (MI300)',
    'gfx942': 'CDNA 3 (MI300)',
    'gfx950': 'CDNA 4 (MI350)',
    'gfx1030': 'RDNA 2 (RX 6000)',
    'gfx1100': 'RDNA 3 (RX 7900)',
    'gfx1101': 'RDNA 3 (RX 7800/7700)',
    'gfx1102': 'RDNA 3 (RX 7600)',
    'gfx1200': 'RDNA 4 (RX 9060)',
    'gfx1201': 'RDNA 4 (RX 9070)',
}


def arch_name(props) -> str:
    gcn = getattr(props, 'gcnArchName', None)
    if IS_ROCM and gcn:
        base = gcn.split(':')[0]
        return f'{AMD_ARCH.get(base, "unknown AMD arch")} [{gcn}]'
    return NVIDIA_ARCH.get((props.major, props.minor), f'unknown (cc {props.major}.{props.minor})')


# ============================================================================
# Section 1: software stack + host
# ============================================================================


def show_stack():
    header('Software Stack')
    print(f'  Python          : {platform.python_version()}  ({sys.executable})')
    print(f'  PyTorch         : {torch.__version__}')
    if IS_ROCM:
        print(f'  Backend         : ROCm / HIP {torch.version.hip}')
    else:
        print(f'  Backend         : CUDA {torch.version.cuda}')
    with contextlib.suppress(Exception):
        print(f'  cuDNN/MIOpen    : {torch.backends.cudnn.version()}')
    try:
        import triton

        print(f'  Triton          : {triton.__version__}  (needed by torch.compile)')
    except ImportError:
        print('  Triton          : NOT INSTALLED — torch.compile will not work')
        flag('Triton missing: torch_compile=true will fail')
    try:
        import flash_attn

        print(f'  flash-attn pkg  : {flash_attn.__version__}')
    except ImportError:
        print('  flash-attn pkg  : not installed (attn_implementation="flash_attention_2" unavailable; sdpa still fine)')

    print(f'\n  Host            : {platform.node()}  ({platform.machine()}, {os.cpu_count()} CPU cores)')
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f'  Host RAM        : {vm.total / GB:.0f} GB total, {vm.available / GB:.0f} GB available')
    except ImportError:
        mem = _sh("free -g | awk '/^Mem:/ {print $2, $7}'")
        if mem:
            tot, avail = mem.split()
            print(f'  Host RAM        : {tot} GB total, {avail} GB available')
    du = shutil.disk_usage('.')
    print(f'  Disk (cwd)      : {du.free / GB:.0f} GB free of {du.total / GB:.0f} GB')


# ============================================================================
# Section 2: GPU identity
# ============================================================================


def show_identity():
    header('GPU Identity')
    n = torch.cuda.device_count()
    print(f'  Visible GPUs    : {n}')

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_by_others = total_mem - free_mem
        smi = smi_gpu_info(i)

        print(f'\n  --- GPU {i}: {props.name} ---')
        print(f'  Architecture       : {arch_name(props)}')
        print(f'  VRAM               : {total_mem / GB:.2f} GB total, {free_mem / GB:.2f} GB free')
        if used_by_others > 0.5 * GB:
            flag(f'GPU {i}: {used_by_others / GB:.2f} GB VRAM already in use by other processes')
        print(f'  Compute units      : {props.multi_processor_count} {"CUs" if IS_ROCM else "SMs"}')
        print(f'  Warp/wavefront     : {props.warp_size}')
        print(f'  L2 cache           : {props.L2_cache_size / 1024**2:.0f} MB')
        if smi.get('driver_version'):
            print(f'  Driver             : {smi["driver_version"]}')
        if smi.get('power.limit'):
            draw = f' (drawing {smi["power.draw"]})' if smi.get('power.draw') else ''
            print(f'  Power limit        : {smi["power.limit"]} W{draw}')
        if smi.get('temperature.gpu'):
            print(f'  Temperature        : {smi["temperature.gpu"]} °C')
        if smi.get('pcie'):
            mx = f'  (max {smi["pcie_max"]})' if smi.get('pcie_max') else ''
            print(f'  PCIe link          : {smi["pcie"]}{mx}')
        if smi.get('ecc.mode.current'):
            print(f'  ECC                : {smi["ecc.mode.current"]}')

        VERDICT.setdefault('gpus', []).append(props.name)
        if i == 0:
            VERDICT['vram_total'] = total_mem / GB
            VERDICT['vram_free'] = free_mem / GB


# ============================================================================
# Section 3: precision & attention support (empirical)
# ============================================================================


def _matmul_ms(dtype, n=2048, iters=10):
    a = torch.randn(n, n, device='cuda', dtype=dtype)
    b = torch.randn(n, n, device='cuda', dtype=dtype)
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    return _event_timer(lambda: torch.mm(a, b), iters=iters)


def show_precision():
    header('Precision Support (measured, not assumed)')

    # fp32 baseline for speedup ratios
    fp32_ms = _matmul_ms(torch.float32)
    n = 2048
    print(f'  matmul {n}x{n}, speedup vs FP32 ({2 * n**3 / (fp32_ms * 1e-3) / 1e12:.1f} TFLOPS):\n')

    # Reference result for numerical-correctness checks
    torch.manual_seed(0)
    a32 = torch.randn(512, 512, device='cuda')
    b32 = torch.randn(512, 512, device='cuda')
    ref = (a32.double() @ b32.double()).float()

    results = {}
    for label, dtype, expect_err in [('FP16', torch.float16, 5e-3), ('BF16', torch.bfloat16, 5e-2)]:
        try:
            ms = _matmul_ms(dtype)
            err = ((a32.to(dtype) @ b32.to(dtype)).float() - ref).norm() / ref.norm()
            speedup = fp32_ms / ms
            accel = 'hardware-accelerated' if speedup > 1.5 else 'NOT accelerated (runs at ~FP32 speed)'
            numerics = 'numerics OK' if err < expect_err * 10 else f'NUMERICS SUSPECT (rel err {err:.1e})'
            print(f'  {label} : {speedup:4.1f}x   {accel}, {numerics}')
            results[label] = speedup
            if err >= expect_err * 10:
                flag(f'{label} matmul relative error {err:.1e} — possibly broken kernels on this stack')
        except Exception as e:
            print(f'  {label} : unavailable ({e})')
            results[label] = 0.0

    # TF32 (NVIDIA Ampere+ only; a no-op path on ROCm)
    if not IS_ROCM:
        try:
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            ms = _matmul_ms(torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = prev
            print(f'  TF32 : {fp32_ms / ms:4.1f}x   (FP32 matmul with allow_tf32=True)')
        except Exception:
            pass

    # FP8 via torch._scaled_mm
    try:
        e4m3 = torch.float8_e4m3fn
        a8 = torch.randn(256, 256, device='cuda').to(e4m3)
        b8 = torch.randn(256, 256, device='cuda').to(e4m3).t().contiguous().t()
        one = torch.tensor(1.0, device='cuda')
        torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print('  FP8  : works  (torch._scaled_mm, e4m3)')
    except Exception:
        print('  FP8  : not supported')

    # INT8 via torch._int_mm
    try:
        a8 = torch.randint(-128, 127, (1024, 1024), device='cuda', dtype=torch.int8)
        b8 = torch.randint(-128, 127, (1024, 1024), device='cuda', dtype=torch.int8)
        torch._int_mm(a8, b8)
        torch.cuda.synchronize()
        print('  INT8 : works  (torch._int_mm)')
    except Exception:
        print('  INT8 : not supported')

    print(f'\n  torch.cuda.is_bf16_supported() : {torch.cuda.is_bf16_supported()}')
    if results.get('BF16', 0) > 1.5 and torch.cuda.is_bf16_supported():
        VERDICT['precision'] = 'bf16'
    elif results.get('FP16', 0) > 1.5:
        VERDICT['precision'] = 'fp16'
        flag('BF16 not hardware-accelerated: prefer fp16 (with grad scaling) on this GPU')
    else:
        VERDICT['precision'] = 'fp32'
        flag('No accelerated half precision detected — training will run at FP32 speed')

    # ── SDPA backends: actually run forward + backward through each ─────────
    print('\n  Attention backends (forward+backward actually executed):')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=dtype, requires_grad=True)
    kv = lambda: (torch.randn_like(q), torch.randn_like(q))
    working = []
    for name, backend in [
        ('flash', SDPBackend.FLASH_ATTENTION),
        ('mem_efficient', SDPBackend.EFFICIENT_ATTENTION),
        ('cudnn', getattr(SDPBackend, 'CUDNN_ATTENTION', None)),
        ('math (fallback)', SDPBackend.MATH),
    ]:
        if backend is None:
            continue
        try:
            k, v = kv()
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                out.sum().backward()
            q.grad = None
            torch.cuda.synchronize()
            print(f'    {name:16s} : works')
            working.append(name)
        except Exception as e:
            print(f'    {name:16s} : NOT available ({str(e).splitlines()[0][:80]})')
    VERDICT['sdpa'] = working
    if 'flash' not in working and 'mem_efficient' not in working:
        flag('No fused attention backend available — attention will be slow and memory-hungry')


# ============================================================================
# Section 4: how much VRAM is actually allocatable
# ============================================================================


def show_allocatable():
    header('VRAM Allocation Test (how much can PyTorch really get?)')
    free0, total = torch.cuda.mem_get_info()
    chunk = 256 * 1024**2  # 256 MB
    blocks = []
    try:
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free < chunk + 512 * 1024**2:  # keep a small reserve
                break
            blocks.append(torch.empty(chunk // 2, dtype=torch.float16, device='cuda'))
    except torch.cuda.OutOfMemoryError:
        pass
    got = len(blocks) * chunk
    blocks.clear()
    torch.cuda.empty_cache()
    pct = 100 * got / total
    print(f'  Allocated {got / GB:.2f} GB of {total / GB:.2f} GB total ({pct:.0f}%) in 256 MB chunks')
    print(f'  (free before test: {free0 / GB:.2f} GB — the rest is driver/display/other processes)')
    VERDICT['vram_usable'] = got / GB
    if pct < 80:
        flag(f'Only {pct:.0f}% of VRAM is allocatable — shared GPU or big reserved carve-out?')


# ============================================================================
# Section 5: benchmarks
# ============================================================================


def bench_tflops():
    header('Compute Throughput (matmul n=4096, 20 iters)')
    n = 4096
    candidates = [('FP32', torch.float32), ('FP16', torch.float16)]
    if torch.cuda.is_bf16_supported():
        candidates.append(('BF16', torch.bfloat16))
    for label, dtype in candidates:
        try:
            ms = _matmul_ms(dtype, n=n, iters=20)
            tflops = 2 * n**3 / (ms * 1e-3) / 1e12
            print(f'  {label} : {tflops:7.2f} TFLOPS  ({ms:.2f} ms/iter)')
            VERDICT[f'tflops_{label.lower()}'] = tflops
        except Exception as e:
            print(f'  {label} : skipped ({e})')


def bench_bandwidth():
    header('Memory Bandwidth (device-to-device copy, 512 MB)')
    n = 512 * 1024**2 // 4
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    ms = _event_timer(lambda: y.copy_(x), iters=10)
    bw = 2 * 512 / (ms * 1e-3) / 1e3  # read + write, GB/s
    print(f'  VRAM bandwidth : {bw:.0f} GB/s')
    VERDICT['bandwidth'] = bw
    x = y = None
    torch.cuda.empty_cache()


def bench_pcie():
    header('PCIe Transfer (pinned host<->device, 256 MB)')
    size_mb = 256
    n = size_mb * 1024**2 // 4
    x_cpu = torch.randn(n).pin_memory()
    x_gpu = torch.empty(n, device='cuda')
    transfers = [
        ('Host→GPU', lambda: x_gpu.copy_(x_cpu, non_blocking=True)),
        ('GPU→Host', lambda: x_cpu.copy_(x_gpu, non_blocking=True)),
    ]
    for label, fn in transfers:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        bw = size_mb * 10 / (time.perf_counter() - t0) / 1024
        print(f'  {label} : {bw:5.2f} GB/s')
        if bw < 3:
            flag(f'PCIe {label} only {bw:.1f} GB/s — data loading to GPU may bottleneck (x1/x4 link?)')
    x_cpu = x_gpu = None
    torch.cuda.empty_cache()


def bench_sdpa():
    header('Attention Speed (per SDPA backend, bsz=4 heads=8 dim=64)')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for seq in (1024, 4096):
        q = torch.randn(4, 8, seq, 64, device='cuda', dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        line = f'  seq={seq:5d} : '
        for name, backend in [
            ('flash', SDPBackend.FLASH_ATTENTION),
            ('mem_eff', SDPBackend.EFFICIENT_ATTENTION),
            ('math', SDPBackend.MATH),
        ]:
            try:
                with sdpa_kernel([backend]):
                    for _ in range(3):
                        F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                    ms = _event_timer(lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=True), iters=10)
                line += f'{name}={ms:6.2f}ms  '
            except Exception:
                line += f'{name}=  n/a   '
        print(line)
        q = k = v = None
    torch.cuda.empty_cache()


def bench_optimizers():
    header('Optimizer Kernels (AdamW variants on a 25M-param tensor)')
    p = torch.randn(25_000_000, device='cuda', requires_grad=True)
    p.grad = torch.randn_like(p)
    for label, kwargs in [('adamw_torch', {}), ('adamw fused', {'fused': True}), ('adamw foreach', {'foreach': True})]:
        try:
            opt = torch.optim.AdamW([p], lr=1e-4, **kwargs)
            opt.step()
            torch.cuda.synchronize()
            ms = _event_timer(opt.step, iters=10)
            print(f'  {label:14s} : {ms:6.2f} ms/step')
            if 'fused' in label:
                VERDICT['fused_adamw'] = True
        except Exception as e:
            print(f'  {label:14s} : FAILS ({str(e).splitlines()[0][:70]})')
            if 'fused' in label:
                VERDICT['fused_adamw'] = False
                flag('Fused AdamW not working: use optim = "adamw_torch" in configs')
    del p
    torch.cuda.empty_cache()


def bench_train_step():
    header('Training Step Simulation (2-layer transformer block, fwd+bwd+AdamW)')
    torch.manual_seed(0)
    batch, seq, d = 8, 512, 768

    candidates = [('FP32', torch.float32), ('FP16', torch.float16)]
    if torch.cuda.is_bf16_supported():
        candidates.append(('BF16', torch.bfloat16))

    for label, amp_dtype in candidates:
        try:
            model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, 8, d * 4, batch_first=True, dropout=0.0),
                num_layers=2,
            ).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            use_amp = amp_dtype != torch.float32
            scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
            x = torch.randn(batch, seq, d, device='cuda')

            def step(model=model, optimizer=optimizer, scaler=scaler, x=x, amp_dtype=amp_dtype, use_amp=use_amp):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    y = model(x)
                    loss = y.float().pow(2).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            ms = _event_timer(step, iters=10)
            print(f'  {label} : {ms:6.2f} ms/step')
            model = optimizer = x = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  {label} : skipped ({e})')


# ============================================================================
# Section 6: torch.compile smoke test
# ============================================================================


def probe_compile():
    header('torch.compile Smoke Test (inductor backend)')
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256)).cuda()
        x = torch.randn(64, 256, device='cuda')
        with torch.no_grad():
            ref = model(x).clone()

        compiled = torch.compile(model)
        t0 = time.perf_counter()
        out = compiled(x)
        out.sum().backward()
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0

        max_diff = (out.detach() - ref).abs().max().item()
        ok = max_diff < 1e-2
        print(f'  compile+first step : {compile_s:.1f} s')
        print(f'  output vs eager    : max diff {max_diff:.2e} {"(OK)" if ok else "(MISMATCH!)"}')
        ms_e = _event_timer(lambda: model(x), iters=20)
        ms_c = _event_timer(lambda: compiled(x), iters=20)
        print(f'  fwd speed          : eager {ms_e:.3f} ms → compiled {ms_c:.3f} ms')
        VERDICT['compile'] = ok
        if not ok:
            flag('torch.compile produces WRONG results on this stack — keep torch_compile=false')
        print('  NOTE: this only proves the compiler stack works. Whether YOUR model')
        print('        trains stably under compile is checked by doctor.py.')
    except Exception as e:
        VERDICT['compile'] = False
        print(f'  torch.compile FAILED: {str(e).splitlines()[0][:100]}')
        flag('torch.compile broken on this stack — set torch_compile=false')
    finally:
        with contextlib.suppress(Exception):
            torch._dynamo.reset()
        torch.cuda.empty_cache()


# ============================================================================
# Section 7: sustained load (thermal throttling)
# ============================================================================


def probe_sustained(seconds):
    header(f'Sustained Load ({seconds} s of continuous matmul — watch for throttling)')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    n = 4096
    a = torch.randn(n, n, device='cuda', dtype=dtype)
    b = torch.randn(n, n, device='cuda', dtype=dtype)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Calibrate how many iterations fit in ~1 second of GPU time, then take
    # one throughput sample per second (kernel launches are async, so a plain
    # wall-clock inner loop would queue far more work than it measures).
    ms = _event_timer(lambda: torch.mm(a, b), iters=10)
    iters = max(1, int(1000 / ms))
    samples = []
    for _ in range(max(1, seconds)):
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        samples.append(2 * n**3 * iters / dt / 1e12)

    first = samples[0]
    worst = min(samples)
    drop = 100 * (first - worst) / first
    print(f'  TFLOPS over time : {" ".join(f"{s:.0f}" for s in samples)}')
    print(f'  first second     : {first:.1f} TFLOPS')
    print(f'  worst second     : {worst:.1f} TFLOPS  ({100 * (worst - first) / first:+.1f}% vs first)')
    smi = smi_gpu_info(0)
    if smi.get('temperature.gpu'):
        print(f'  temp after burn  : {smi["temperature.gpu"]} °C')
    if smi.get('power.draw'):
        print(f'  power draw       : {smi["power.draw"]} W (limit {smi.get("power.limit", "?")} W)')
    if drop > 15:
        flag(f'Sustained throughput drops {drop:.0f}% — thermal/power throttling; long runs will be slower than benchmarks')
    a = b = None
    torch.cuda.empty_cache()


# ============================================================================
# Section 8: multi-GPU
# ============================================================================


def show_multi_gpu():
    n = torch.cuda.device_count()
    if n < 2:
        return
    header('Multi-GPU Peer-to-Peer')
    for i in range(n):
        for j in range(n):
            if i != j:
                ok = torch.cuda.can_device_access_peer(i, j)
                line = f'  GPU{i}→GPU{j} : P2P {"yes" if ok else "no "}'
                try:
                    src = torch.randn(64 * 1024**2 // 4, device=f'cuda:{i}')
                    dst = torch.empty_like(src, device=f'cuda:{j}')
                    dst.copy_(src)
                    torch.cuda.synchronize(i)
                    torch.cuda.synchronize(j)
                    t0 = time.perf_counter()
                    for _ in range(10):
                        dst.copy_(src)
                    torch.cuda.synchronize(i)
                    torch.cuda.synchronize(j)
                    bw = 64 * 10 / (time.perf_counter() - t0) / 1024
                    line += f'  copy {bw:.1f} GB/s'
                    del src, dst
                except Exception:
                    pass
                print(line)
    torch.cuda.empty_cache()


# ============================================================================
# Verdict
# ============================================================================


def show_verdict():
    header('VERDICT')
    gpus = VERDICT.get('gpus', [])
    print(f'  GPU              : {len(gpus)}x {gpus[0] if gpus else "?"}')
    print(f'  Platform         : {"ROCm " + str(torch.version.hip) if IS_ROCM else "CUDA " + str(torch.version.cuda)}')
    if 'vram_usable' in VERDICT:
        print(f'  Usable VRAM      : {VERDICT["vram_usable"]:.1f} GB of {VERDICT.get("vram_total", 0):.1f} GB')
    if 'tflops_bf16' in VERDICT or 'tflops_fp16' in VERDICT:
        tf = VERDICT.get('tflops_bf16') or VERDICT.get('tflops_fp16')
        print(f'  Half-prec compute: {tf:.0f} TFLOPS measured')
    if 'bandwidth' in VERDICT:
        print(f'  VRAM bandwidth   : {VERDICT["bandwidth"]:.0f} GB/s measured')
    print(
        f'  Recommended prec : {VERDICT.get("precision", "?")}  '
        f'{"(bf16 = true)" if VERDICT.get("precision") == "bf16" else "(fp16 = true)" if VERDICT.get("precision") == "fp16" else ""}'
    )
    sdpa = VERDICT.get('sdpa', [])
    best = 'flash' if 'flash' in sdpa else ('mem_efficient' if 'mem_efficient' in sdpa else 'math only')
    print(f'  Attention        : sdpa with {best} backend')
    if 'fused_adamw' in VERDICT:
        print(f'  Fused AdamW      : {"works" if VERDICT["fused_adamw"] else "BROKEN — use adamw_torch"}')
    if 'compile' in VERDICT:
        compile_msg = 'stack works (validate YOUR model with doctor.py)' if VERDICT['compile'] else 'BROKEN — torch_compile = false'
        print(f'  torch.compile    : {compile_msg}')

    if FLAGS:
        print(f'\n  Red flags ({len(FLAGS)}):')
        for f in FLAGS:
            print(f'    🚩 {f}')
    else:
        print('\n  No red flags. 🎉')
    print(SEP)


def main():
    parser = argparse.ArgumentParser(description='Standalone GPU capability probe')
    parser.add_argument('--quick', action='store_true', help='identity + feature detection only (skip benchmarks)')
    parser.add_argument('--burn', type=int, default=10, help='seconds of sustained-load test (0 to skip, default 10)')
    args = parser.parse_args()

    show_stack()
    if not torch.cuda.is_available():
        print(f'\n{SEP}')
        print('  ❌ No GPU visible to PyTorch (torch.cuda.is_available() = False)')
        if IS_ROCM:
            print('     ROCm build detected — check that the GPU arch is supported (HSA_OVERRIDE_GFX_VERSION?)')
        else:
            print('     CUDA build detected — check drivers / CUDA_VISIBLE_DEVICES')
        print(SEP)
        raise SystemExit(1)

    show_identity()
    show_precision()

    if not args.quick:
        show_allocatable()
        bench_tflops()
        bench_bandwidth()
        bench_pcie()
        bench_sdpa()
        bench_optimizers()
        bench_train_step()
        probe_compile()
        if args.burn > 0:
            probe_sustained(args.burn)
        show_multi_gpu()

    show_verdict()


if __name__ == '__main__':
    main()
