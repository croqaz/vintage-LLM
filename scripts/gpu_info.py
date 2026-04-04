import torch
from lightning.fabric.utilities.throughput import measure_flops
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy


def display_gpu_info():
    print('=' * 40)
    print('GPU Information:')
    print('=' * 40)
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
        return False

    num_devices = torch.cuda.device_count()
    print(f'Number of GPUs: {num_devices}')

    for i in range(num_devices):
        print(f'\n--- GPU {i} ---')
        props = torch.cuda.get_device_properties(i)
        print(f'Model: {props.name}')
        print(f'Architecture: Compute Capability {props.major}.{props.minor}')

        # Memory Info
        total_mem = props.total_memory / (1024**3)  # Convert to GB
        free_mem, _ = torch.cuda.mem_get_info(i)
        free_mem = free_mem / (1024**3)
        print(f'Total Memory: {total_mem:.2f} GB')
        print(f'Available Memory: {free_mem:.2f} GB')

        # Multiprocessors / Cores info
        print(f'Streaming Multiprocessors (SMs): {props.multi_processor_count}')

    print('=' * 40)
    return True


def calculate_measured_tflops(model_name='Llama-2-7b-hf'):
    print(f'\nCalculating Theoretical TFLOPs for a forward+backward pass of {model_name}...')

    # Initialize config
    try:
        config = Config.from_name(model_name)
    except ValueError as e:
        print(f'Error loading config: {e}')
        return

    # Typical small micro batch size and max sequence length for checking
    micro_batch_size = 1
    seq_length = config.block_size if getattr(config, 'block_size', None) else 2048

    # We use meta device to not allocate actual memory for weights, just to count FLOPs
    with torch.device('meta'):
        meta_model = GPT(config)

        # Dummy inputs for FLOPs calculation
        x = torch.randint(0, config.vocab_size, (micro_batch_size, seq_length))

        def model_fwd():
            return meta_model(x)

        def model_loss(y):
            # Same logic used in pretrain.py
            return chunked_cross_entropy(y, x, chunk_size=0)

        # calculate flops. `measure_flops` traces operations from Lightning Fabric
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)

    tflops_per_step = measured_flops / 1e12
    print(f'Measured TFLOPs per step (batch_size={micro_batch_size}, seq_len={seq_length}): {tflops_per_step:.2f}')


if __name__ == '__main__':
    display_gpu_info()
    calculate_measured_tflops('Llama-3-8B')
