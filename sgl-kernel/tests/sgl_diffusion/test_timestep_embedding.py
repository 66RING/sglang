import numpy as np
import pytest
import tabulate
import torch
from sgl_kernel.elementwise import timestep_embedding as timestep_embedding_cuda

from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 11008, 16384])
@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 8192])
def test_timestep_embedding(batch_size, dim):
    device = "cuda"
    t = torch.randn((batch_size,), device=device, dtype=torch.float32)
    torch_output = timestep_embedding(t, dim)
    cuda_output = timestep_embedding_cuda(t, dim)
    torch.testing.assert_close(torch_output, cuda_output + 1)


def test_timestep_embedding_perf():
    NUM_BATCH = [1, 2, 8, 63, 256, 512, 613, 1024, 1536]
    NUM_DIM = [32, 64, 128, 256, 259, 512, 613, 1024, 2048, 4096]

    def perf_kernel_fn(kernel_fn: callable, *args, **kwargs):
        warmup_times = 4
        repeat_times = 20
        total_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup_times):
            output_fn = kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()

        for _ in range(repeat_times):
            start.record()
            output_fn = kernel_fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

        return total_time / repeat_times

    # TODO: hard code
    device = "cuda"
    results = []

    cuda_speedups = []
    for batch_size in NUM_BATCH:
        for dim in NUM_DIM:
            t = torch.randn((batch_size,), device=device)

            time_torch = perf_kernel_fn(timestep_embedding, t, dim)
            time_cuda = perf_kernel_fn(timestep_embedding_cuda, t, dim)
            speedup_cuda = time_torch / time_cuda

            results.append(
                {
                    "Batch Size": batch_size,
                    "Dimension": dim,
                    "Torch Time (ms)": time_torch,
                    "CUDA Time (ms)": time_cuda,
                    "Speedup (CUDA)": speedup_cuda,
                }
            )
            cuda_speedups.append(speedup_cuda)

    print("=== Timestep Embedding Benchmark Results ===")
    print(
        tabulate.tabulate(
            results,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt=(".0f", ".0f", ".6f", ".6f", ".5f"),
        )
    )
    print(f"Average Speedup(cuda): {np.mean(cuda_speedups):.4f}")


if __name__ == "__main__":
    pytest.main([__file__])
