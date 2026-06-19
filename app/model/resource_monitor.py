"""
resource_monitor.py
Background sampler for this process's CPU, RAM, and GPU usage during a
block of work (e.g. one inference run), producing a written .txt report.

Caveats
-------
- CPU% and RAM (RSS) are exact to this process (via psutil) — other
  processes on the machine do not affect these numbers.
- VRAM allocated/reserved are exact to this process (via torch.cuda) since
  each process owns its own CUDA context.
- GPU utilization (% of SM busy) is reported device-wide by NVIDIA's driver.
  There is no per-process compute-utilization metric exposed by pynvml or
  nvidia-smi, so this figure includes any other process using the same GPU.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


@dataclass
class _Sample:
    cpu_pct: float
    rss_mb: float
    gpu_util_pct: Optional[float]
    gpu_mem_used_mb: Optional[float]


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _peak(values):
    return max(values) if values else 0.0


class ResourceMonitor:
    """
    Usage:
        mon = ResourceMonitor()
        mon.start()
        ... do work ...
        report_text = mon.stop(label="inference")
    """

    def __init__(self, interval_s: float = 0.3):
        self._interval = interval_s
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._samples: list[_Sample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

        self._nvml_handle = None
        if _NVML_OK:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    def start(self):
        if self._proc:
            self._proc.cpu_percent(None)  # prime psutil's internal counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            self._samples.append(self._sample())
            self._stop_event.wait(self._interval)

    def _sample(self) -> _Sample:
        cpu_pct = self._proc.cpu_percent(None) if self._proc else 0.0
        rss_mb  = self._proc.memory_info().rss / 1e6 if self._proc else 0.0

        gpu_util = gpu_mem = None
        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_util = float(util.gpu)
                gpu_mem  = mem.used / 1e6
            except Exception:
                pass

        return _Sample(cpu_pct, rss_mb, gpu_util, gpu_mem)

    def stop(self, label: str = "inference", out_dir: Optional[str] = None) -> str:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        elapsed = time.time() - self._t0

        peak_vram_alloc_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        )
        peak_vram_reserved_mb = (
            torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else 0.0
        )

        cpu_vals  = [s.cpu_pct for s in self._samples]
        rss_vals  = [s.rss_mb for s in self._samples]
        gpu_utils = [s.gpu_util_pct for s in self._samples if s.gpu_util_pct is not None]
        gpu_mems  = [s.gpu_mem_used_mb for s in self._samples if s.gpu_mem_used_mb is not None]

        lines = [
            f"=== Resource report: {label} ===",
            f"Wall-clock time:        {elapsed:.2f} s",
            f"Samples collected:      {len(self._samples)}",
            "",
        ]

        if psutil:
            lines += [
                "-- This process (psutil, exact to this PID) --",
                f"CPU avg / peak:          {_avg(cpu_vals):.1f}% / {_peak(cpu_vals):.1f}%  "
                f"(100% = 1 full core; can exceed 100% across multiple cores)",
                f"RAM (RSS) avg / peak:    {_avg(rss_vals):.1f} MB / {_peak(rss_vals):.1f} MB",
                "",
            ]
        else:
            lines += ["-- psutil not installed: CPU/RAM figures skipped --", ""]

        if torch.cuda.is_available():
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
            lines += [
                "-- GPU memory (torch, exact to this process) --",
                f"VRAM allocated peak:     {peak_vram_alloc_mb:.1f} MB",
                f"VRAM reserved peak:      {peak_vram_reserved_mb:.1f} MB  "
                f"(includes PyTorch's cached/unused blocks)",
                f"Total device VRAM:       {total_vram_mb:.1f} MB",
                f"Peak / total:            {100 * peak_vram_alloc_mb / total_vram_mb:.1f}%",
                "",
            ]
        else:
            lines += ["-- GPU: CUDA not available, this ran on CPU --", ""]

        if _NVML_OK and gpu_utils:
            lines += [
                "-- GPU utilization (device-wide — NVIDIA exposes no per-process "
                "compute-utilization metric, so this includes any other process "
                "using the same GPU) --",
                f"GPU util avg / peak:     {_avg(gpu_utils):.1f}% / {_peak(gpu_utils):.1f}%",
                f"GPU mem used avg / peak: {_avg(gpu_mems):.1f} MB / {_peak(gpu_mems):.1f} MB",
            ]
        else:
            lines += ["-- GPU utilization: pynvml not installed, skipped --"]

        report = "\n".join(lines)

        out_dir = Path(out_dir) if out_dir else Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"resource_report_{label}_{ts}.txt"
        report_path.write_text(report, encoding="utf-8")

        return report
