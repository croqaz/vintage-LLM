import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SEP = '=' * 60


def _event_timer(fn, iters=10):
    """Returns average elapsed ms using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def display_gpu_info():
    print(SEP)
    print('  GPU Information')
    print(SEP)
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
        return False

    print(f'PyTorch version : {torch.__version__}')
    print(f'CUDA version    : {torch.version.cuda}')
    print(f'cuDNN version   : {torch.backends.cudnn.version()}')
    print(f'Number of GPUs  : {torch.cuda.device_count()}')

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f'\n--- GPU {i}: {props.name} ---')
        print(f'  Compute Capability  : {props.major}.{props.minor}')
        print(f'  Total Memory        : {props.total_memory / 1024**3:.2f} GB')
        print(f'  Free Memory         : {free_mem / 1024**3:.2f} GB')
        print(f'  Streaming MPs (SMs) : {props.multi_processor_count}')
        print(f'  L2 Cache            : {props.L2_cache_size / 1024**2:.0f} MB')
        print(f'  Max threads/block   : {props.max_threads_per_block}')
        print(f'  Warp size           : {props.warp_size}')
        print(f'  Shared mem/SM       : {props.shared_memory_per_block / 1024:.0f} KB')

    return True


def check_precision_support():
    print(f'\n{SEP}')
    print('  Precision & Feature Support')
    print(SEP)

    cap = torch.cuda.get_device_capability()

    def yn(v):
        return 'yes' if v else 'no'

    print(f'  FP16 Tensor Cores (Volta+  cc>=7.0) : {yn(cap >= (7, 0))}')
    print(f'  BF16 / TF32       (Ampere+ cc>=8.0) : {yn(cap >= (8, 0))}')
    print(f'  FP8               (Hopper+ cc>=9.0) : {yn(cap >= (9, 0))}')
    print(f'  BF16 supported (torch)              : {yn(torch.cuda.is_bf16_supported())}')
    print(f'  TF32 matmul enabled                 : {yn(torch.backends.cuda.matmul.allow_tf32)}')
    print(f'  TF32 cuDNN enabled                  : {yn(torch.backends.cudnn.allow_tf32)}')
    print(f'  Flash SDP available                 : {yn(torch.backends.cuda.flash_sdp_enabled())}')
    print(f'  Mem-efficient SDP available         : {yn(torch.backends.cuda.mem_efficient_sdp_enabled())}')
    print(f'  Math (fallback) SDP available       : {yn(torch.backends.cuda.math_sdp_enabled())}')


def benchmark_compute_tflops():
    print(f'\n{SEP}')
    print('  Compute Throughput  (matrix multiply, n=4096, 20 iters)')
    print(SEP)

    n = 4096
    cap = torch.cuda.get_device_capability()
    dtypes = [('FP32', torch.float32)]
    if cap >= (7, 0):
        dtypes.append(('FP16', torch.float16))
    if cap >= (8, 0):
        dtypes.append(('BF16', torch.bfloat16))

    for label, dtype in dtypes:
        try:
            a = torch.randn(n, n, device='cuda', dtype=dtype)
            b = torch.randn(n, n, device='cuda', dtype=dtype)
            for _ in range(3):
                torch.mm(a, b)
            torch.cuda.synchronize()
            ms = _event_timer(lambda: torch.mm(a, b), iters=20)
            tflops = 2 * n**3 / (ms * 1e-3) / 1e12
            print(f'  {label:6s} : {tflops:7.2f} TFLOPS  ({ms:.2f} ms/iter)')
        except Exception as e:
            print(f'  {label:6s} : skipped ({e})')

    # INT8 via torch._int_mm if available
    try:
        a8 = torch.randint(-128, 127, (n, n), device='cuda', dtype=torch.int8)
        b8 = torch.randint(-128, 127, (n, n), device='cuda', dtype=torch.int8)
        for _ in range(3):
            torch._int_mm(a8, b8)
        torch.cuda.synchronize()
        ms = _event_timer(lambda: torch._int_mm(a8, b8), iters=20)
        tflops = 2 * n**3 / (ms * 1e-3) / 1e12
        print(f'  {"INT8":6s} : {tflops:7.2f} TFLOPS  ({ms:.2f} ms/iter)')
    except Exception:
        pass


def benchmark_memory_bandwidth():
    print(f'\n{SEP}')
    print('  Memory Bandwidth  (device-to-device copy, 512 MB)')
    print(SEP)

    n = 512 * 1024**2 // 4  # float32 elements → 512 MB
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()

    ms = _event_timer(lambda: y.copy_(x), iters=10)
    bw = 2 * 512 / (ms * 1e-3) / 1e3  # read + write, GB/s
    print(f'  HBM bandwidth : {bw:.1f} GB/s')


def benchmark_pcie_bandwidth():
    print(f'\n{SEP}')
    print('  PCIe Bandwidth  (pinned host↔device, 256 MB)')
    print(SEP)

    size_mb = 256
    n = size_mb * 1024**2 // 4
    x_cpu = torch.randn(n).pin_memory()
    x_gpu = torch.empty(n, device='cuda')

    for label, fn in [('H→D', lambda: x_gpu.copy_(x_cpu)), ('D→H', lambda: x_cpu.copy_(x_gpu))]:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        bw = size_mb * 10 / (time.perf_counter() - t0) / 1024
        print(f'  PCIe {label} : {bw:.2f} GB/s')


def benchmark_attention():
    print(f'\n{SEP}')
    print('  Attention (scaled_dot_product_attention, seq=1024, 10 iters)')
    print(SEP)

    batch, heads, seq, head_dim = 2, 16, 1024, 64
    dtype = torch.float16 if torch.cuda.get_device_capability() >= (7, 0) else torch.float32

    q = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run_sdpa():
        F.scaled_dot_product_attention(q, k, v)

    candidates = [
        ('Flash', [SDPBackend.FLASH_ATTENTION]),
        ('MemEff', [SDPBackend.EFFICIENT_ATTENTION]),
        ('Math', [SDPBackend.MATH]),
    ]

    for name, backends in candidates:
        try:
            with sdpa_kernel(backends):
                for _ in range(3):
                    run_sdpa()
                torch.cuda.synchronize()
                ms = _event_timer(run_sdpa, iters=10)
            print(f'  {name:8s} SDPA : {ms:.2f} ms/iter')
        except Exception as e:
            print(f'  {name:8s} SDPA : unavailable ({e})')


def benchmark_training_step():
    print(f'\n{SEP}')
    print('  Training Step Simulation  (fwd + bwd + AdamW, 10 iters each)')
    print(f'  batch={4}, seq={256}, d={512}')
    print(SEP)

    batch, seq, d = 4, 256, 512
    x = torch.randn(batch, seq, d, device='cuda')

    cap = torch.cuda.get_device_capability()
    candidates = [('FP32', torch.float32)]
    if cap >= (7, 0):
        candidates.append(('FP16', torch.float16))
    if torch.cuda.is_bf16_supported():
        candidates.append(('BF16', torch.bfloat16))

    for label, amp_dtype in candidates:
        try:
            model = nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Linear(d * 4, d),
            ).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            use_amp = amp_dtype != torch.float32
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

            for _ in range(3):  # warm-up
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    y = model(x)
                scaler.scale(y.sum()).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.cuda.synchronize()

            def step():
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    y = model(x)
                scaler.scale(y.sum()).backward()
                scaler.step(optimizer)
                scaler.update()

            ms = _event_timer(step, iters=10)
            print(f'  {label:4s} : {ms:.2f} ms/step')
        except Exception as e:
            print(f'  {label:4s} : skipped ({e})')


def check_multi_gpu():
    n = torch.cuda.device_count()
    if n < 2:
        return

    print(f'\n{SEP}')
    print('  Multi-GPU P2P Access')
    print(SEP)
    for i in range(n):
        for j in range(n):
            if i != j:
                ok = torch.cuda.can_device_access_peer(i, j)
                print(f'  P2P GPU{i}→GPU{j} : {"yes" if ok else "no"}')


def memory_stats():
    print(f'\n{SEP}')
    print('  Memory Stats (post-benchmarks)')
    print(SEP)
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.max_memory_allocated(i) / 1024**3
        reserved = torch.cuda.max_memory_reserved(i) / 1024**3
        free, total = torch.cuda.mem_get_info(i)
        print(f'  GPU {i}:')
        print(f'    Peak allocated : {alloc:.3f} GB')
        print(f'    Peak reserved  : {reserved:.3f} GB')
        print(f'    Currently free : {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB')


if __name__ == '__main__':
    if not display_gpu_info():
        raise SystemExit(1)

    check_precision_support()
    benchmark_compute_tflops()
    benchmark_memory_bandwidth()
    benchmark_pcie_bandwidth()
    benchmark_attention()
    benchmark_training_step()
    check_multi_gpu()
    memory_stats()

    print(f'\n{SEP}')
    print('  Done.')
    print(SEP)
