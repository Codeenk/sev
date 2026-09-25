"""The accelerator this process uses: cuda, then mps, then cpu."""
import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _kind(device):
    return str(device).split(":")[0]


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if _kind(device) == "mps": torch.mps.synchronize()
    elif _kind(device) == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if _kind(device) == "mps": torch.mps.empty_cache()
    elif _kind(device) == "cuda": torch.cuda.empty_cache()


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if _kind(device) == "mps": return torch.mps.current_allocated_memory()
    if _kind(device) == "cuda": return torch.cuda.max_memory_allocated()
    return 0
