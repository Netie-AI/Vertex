"""
Netie AI — TensorRT Compilation Script
======================================
Converts ONNX models to optimized TensorRT engines for Edge IPCs.
"""

import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("netie_ai.compile_trt")

try:
    import tensorrt as trt
except ImportError:
    trt = None

def build_engine(onnx_path, output_path, precision="fp16", workspace_gb=4.0):
    if trt is None:
        raise ImportError("TensorRT required for compilation.")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    logger.info("Parsing ONNX: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Using FP16 precision")

    logger.info("Building Engine... (this may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(serialized_engine)
    logger.info("Engine saved: %s", output)

def main():
    parser = argparse.ArgumentParser(description="Compile ONNX to TensorRT")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    args = parser.parse_args()
    build_engine(args.onnx, args.output, args.precision)

if __name__ == "__main__":
    main()
