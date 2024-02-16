__all__ = ["TorchTimeProfiler", "TorchMemoryProfiler"]

import gc
import logging
import os
import pickle

try:
    import torch
    from torch.cuda._memory_viz import trace_plot
    _IMPORTED_TORCH = True
except ImportError:
    _IMPORTED_TORCH = False

logger = logging.getLogger(__name__)

class TorchTimeProfiler(torch.profiler.profile):
    def __init__(self, **kwargs):
        super().__init__(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], **kwargs)

    def cpu_time(self, row_limit=10):
        return self.key_averages().table(sort_by="cpu_time", row_limit=row_limit, top_level_events_only=True)
    def cpu_time_all(self, row_limit=10):
        return self.key_averages().table(sort_by="cpu_time", row_limit=row_limit, top_level_events_only=False)

    def cpu_time_total(self, row_limit=10):
        return self.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit, top_level_events_only=True)
    def cpu_time_total_all(self, row_limit=10):
        return self.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit, top_level_events_only=False)

    def cuda_time(self, row_limit=10):
        return self.key_averages().table(sort_by="cuda_time", row_limit=row_limit, top_level_events_only=True)
    def cuda_time_all(self, row_limit=10):
        return self.key_averages().table(sort_by="cuda_time", row_limit=row_limit, top_level_events_only=False)

    def cuda_time_total(self, row_limit=10):
        return self.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit, top_level_events_only=True)
    def cuda_time_total_all(self, row_limit=10):
        return self.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit, top_level_events_only=False)


class TorchMemoryProfiler():
    """Class for profiling GPU memory usage by PyTorch."""

    def __init__(self, reset_peak_stats=False, record_mem_history=False):
        self.reset_peak_stats = reset_peak_stats
        self.record_mem_history = record_mem_history

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        gc.collect()
        if self.reset_peak_stats:
            torch.cuda.memory.reset_peak_memory_stats()
        if self.record_mem_history:
            TorchMemoryProfiler.record_memory_history()

    def stop(self):
        self.gpu_mem_stats = torch.cuda.memory.memory_stats()
        if self.record_mem_history:
            self.gpu_mem_history = torch.cuda.memory._snapshot()

    def save(self, path="memory_history"):
        if self.record_mem_history:
            TorchMemoryProfiler.save_memory_history(path=path, snapshot=self.gpu_mem_history)

    @property
    def gpu_peak_memory(self):
        return self.gpu_mem_stats.get("allocated_bytes.all.peak", 0)

    @property
    def gpu_peak_memory_reserved(self):
        return self.gpu_mem_stats.get("reserved_bytes.all.peak", 0)

    @staticmethod
    def record_memory_history():
        if torch.cuda.is_available():
            if hasattr(torch.cuda.memory, "_dump_snapshot"):
                # start recording memory history
                torch.cuda.memory._record_memory_history()
            else:
                logger.warning("Recording memory history requires torch >= 2.1.")

    @staticmethod
    def save_memory_history(path="memory_history", snapshot=None):
        if torch.cuda.is_available():
            if hasattr(torch.cuda.memory, "_dump_snapshot"):
                # create snapshot of memory history
                if snapshot is None:
                    snapshot = torch.cuda.memory._snapshot()
                # save snapshot of memory history as pickle and HTML
                if os.path.dirname(path) != "":
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(f"{path}.pickle", "wb") as file:
                    pickle.dump(snapshot, file)
                with open(f"{path}.html", "w") as file:
                    file.write(trace_plot(snapshot))
            else:
                logger.warning("Saving memory history requires torch >= 2.1.")
