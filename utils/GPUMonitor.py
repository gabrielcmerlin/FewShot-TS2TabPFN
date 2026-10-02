import torch

class GPUMemoryMonitor:
    """Monitora o pico de VRAM usando as métricas nativas do PyTorch (sem threads)."""

    def __init__(self):
        # Parâmetro mantido por compatibilidade de assinatura, mas ignorado
        pass

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0