"""
Netie AI — TensorRT Inference Engine Wrapper
===========================================
Loads a serialised TensorRT .engine file and runs synchronous inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("netie_ai.trt_wrapper")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    # Allow non-GPU environments for documentation/testing
    trt = None
    cuda = None


class TRTInferenceEngine:
    """Wraps a TensorRT engine for single-image anomaly inference."""

    def __init__(
        self,
        engine_path: str,
        input_shape: tuple[int, int, int] = (3, 256, 256),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        if trt is None or cuda is None:
            raise ImportError("TensorRT/PyCUDA required for TRTInferenceEngine.")

        self.input_shape = input_shape
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(size)

            self.bindings.append(int(device_mem))
            buf = {"host": host_mem, "device": device_mem, "shape": shape, "name": name}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buf)
                self.context.set_tensor_address(name, int(device_mem))
            else:
                self.outputs.append(buf)
                self.context.set_tensor_address(name, int(device_mem))

        logger.info("Engine loaded: %s", engine_path.name)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        _, h, w = self.input_shape
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = (img - self.mean) / self.std
        return np.ascontiguousarray(np.expand_dims(img, 0))

    def infer(self, frame: np.ndarray) -> tuple[float, np.ndarray]:
        preprocessed = self.preprocess(frame)

        np.copyto(self.inputs[0]["host"], preprocessed.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        anomaly_map = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        anomaly_score_raw = self.outputs[1]["host"].reshape(self.outputs[1]["shape"])

        amap = anomaly_map.squeeze()
        if amap.max() > amap.min():
            amap = (amap - amap.min()) / (amap.max() - amap.min())
        else:
            amap = np.zeros_like(amap)

        return float(anomaly_score_raw.squeeze()), amap

    @staticmethod
    def overlay_heatmap(frame: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        h, w = frame.shape[:2]
        heatmap = cv2.resize(anomaly_map, (w, h))
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    def __del__(self):
        try:
            del self.context
            del self.engine
        except:
            pass
