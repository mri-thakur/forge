"""Pause GPU training while the GPU is too hot.

Hours of training on a laptop can overheat it badly enough for the operating
system to hibernate it, which happened on the RTX 4050 laptop at a GPU temperature
of about 87 °C. The guard reads the temperature through NVML (the library behind
nvidia-smi) and, at or above `limit`, sleeps between steps until the GPU has cooled
to `resume`. Set FORGE_GPU_TEMP_LIMIT to change the limit, or to 0 to disable it.
"""

from __future__ import annotations

import ctypes
import json
import os
import time


class Nvml:
    """Temperature of the first NVIDIA GPU via the driver's NVML library."""

    def __init__(self):
        if os.name == "nt":
            names = ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
        else:
            names = ["libnvidia-ml.so.1"]
        for name in names:
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError("NVML library not found")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("NVML failed to initialize")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise OSError("no NVML device 0")

    def temperature(self):
        value = ctypes.c_uint()
        # 0 = NVML_TEMPERATURE_GPU, the GPU die sensor.
        if self.lib.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(value)) != 0:
            return None
        return value.value


class ThermalGuard:
    def __init__(self, limit, resume, read, sleep=time.sleep, log=print, interval=2.0):
        if resume >= limit:
            raise ValueError("resume temperature must be below the limit")
        self.limit, self.resume, self.read = limit, resume, read
        self.sleep, self.log, self.interval = sleep, log, interval
        self.pauses, self.paused_seconds = 0, 0.0

    @classmethod
    def for_device(cls, device, limit=None):
        """A guard for CUDA training, or None when disabled or unreadable."""
        if limit is None:
            limit = int(os.environ.get("FORGE_GPU_TEMP_LIMIT", "85"))
        if not str(device).startswith("cuda") or limit <= 0:
            return None
        try:
            nvml = Nvml()
        except OSError:
            return None
        return cls(limit, limit - 10, nvml.temperature)

    def wait_if_hot(self):
        """Call while the GPU is idle; returns the seconds spent cooling down."""
        temperature = self.read()
        if temperature is None or temperature < self.limit:
            return 0.0
        start = time.perf_counter()
        self.pauses += 1
        self.log(json.dumps({"thermal_pause": {"gpu_celsius": temperature, "limit": self.limit}}))
        while temperature is not None and temperature > self.resume:
            self.sleep(self.interval)
            temperature = self.read()
        waited = time.perf_counter() - start
        self.paused_seconds += waited
        self.log(json.dumps({"thermal_resume": {"gpu_celsius": temperature, "seconds": waited}}))
        return waited
