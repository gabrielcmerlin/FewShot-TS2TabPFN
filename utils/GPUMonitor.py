import time
import threading
import torch

class GPUMemoryMonitor:
    """Monitora a VRAM (alocada) em background coletando pico e média durante um bloco de código."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            if torch.cuda.is_available():
                self.samples.append(torch.cuda.memory_allocated() / (1024**2))
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Registra o ponto inicial
            self.samples.append(torch.cuda.memory_allocated() / (1024**2))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        # Garante que registramos o pico final caso a thread não tenha pego no milissegundo exato
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_mb = 0.0

        return peak_mb