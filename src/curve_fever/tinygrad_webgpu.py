from __future__ import annotations

import ctypes

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.device import Device

from . import _curve_webgpu_native


def _ptr(obj) -> int:
    return int(ctypes.cast(obj, ctypes.c_void_p).value)


def device_ptr(device: str = "WEBGPU") -> int:
    return _ptr(Device[device].device_res)


def tensor_buffer_ptr(tensor: Tensor) -> int:
    tensor.realize()
    buf = tensor.uop.base.buffer
    buf.ensure_allocated()
    return _ptr(buf._buf)


def rgba(r: int, g: int, b: int, a: int = 255) -> int:
    return int(r) | (int(g) << 8) | (int(b) << 16) | (int(a) << 24)


def initial_batch_players(batch_size: int, width: int, height: int, device: str = "WEBGPU") -> Tensor:
    raw = np.zeros((batch_size, 2, 8), dtype=np.uint32)
    floats = raw.view(np.float32)
    floats[:, 0, 0] = width * 0.25
    floats[:, 0, 1] = height * 0.50
    floats[:, 0, 2] = width * 0.25
    floats[:, 0, 3] = height * 0.50
    floats[:, 0, 4] = 0.0
    raw[:, 0, 5] = 1
    raw[:, 0, 6] = rgba(60, 220, 255)
    floats[:, 1, 0] = width * 0.75
    floats[:, 1, 1] = height * 0.50
    floats[:, 1, 2] = width * 0.75
    floats[:, 1, 3] = height * 0.50
    floats[:, 1, 4] = np.pi
    raw[:, 1, 5] = 1
    raw[:, 1, 6] = rgba(255, 92, 120)
    return Tensor(raw.reshape(-1), dtype=dtypes.uint32, device=device).contiguous().realize()


class BatchBenchmarkBridge:
    """Dispatches the shared C++ batch benchmark shader on tinygrad-owned buffers."""

    def __init__(self, batch_size: int, width: int = 128, height: int = 128, device: str = "WEBGPU"):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.device = device
        self.players = initial_batch_players(batch_size, width, height, device)
        self.occupancy = Tensor.zeros(batch_size * width * height, dtype=dtypes.uint32, device=device).contiguous().realize()
        self.image = Tensor.zeros(batch_size * width * height, dtype=dtypes.uint32, device=device).contiguous().realize()
        self.runner = _curve_webgpu_native.create_batch_runner(device_ptr(device), batch_size, width, height)

    def step(self, frame: int, speed: float = 1.65, turn_rate: float = 0.155, radius: float = 1.75) -> None:
        _curve_webgpu_native.batch_step(
            self.runner,
            int(frame),
            tensor_buffer_ptr(self.players),
            tensor_buffer_ptr(self.occupancy),
            tensor_buffer_ptr(self.image),
            float(speed),
            float(turn_rate),
            float(radius),
        )

    def buffer_sizes(self) -> dict[str, int]:
        return {
            "players": _curve_webgpu_native.buffer_size(tensor_buffer_ptr(self.players)),
            "occupancy": _curve_webgpu_native.buffer_size(tensor_buffer_ptr(self.occupancy)),
            "image": _curve_webgpu_native.buffer_size(tensor_buffer_ptr(self.image)),
        }
