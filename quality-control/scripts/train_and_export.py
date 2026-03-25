"""
Netie AI — Train and Export Pipeline
====================================
Combined pipeline for training an Anomalib PatchCore model and exporting to ONNX/TensorRT.
"""

import argparse
import logging
from pathlib import Path
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("netie_ai.train_export")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(cfg: dict):
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    logger.info("Initializing Datamodule...")
    ds_cfg = cfg["dataset"]
    datamodule = Folder(
        name="netie_inspection",
        root=ds_cfg["root"],
        normal_dir=ds_cfg["normal_dir"],
        abnormal_dir=ds_cfg.get("abnormal_dir"),
        task=ds_cfg.get("task", "segmentation"),
        image_size=tuple(ds_cfg.get("image_size", [256, 256])),
    )

    logger.info("Initializing Model...")
    m_cfg = cfg["model"]
    model = Patchcore(
        backbone=m_cfg.get("backbone", "wide_resnet50_2"),
        layers=m_cfg.get("layers_to_extract", ["layer2", "layer3"]),
    )

    logger.info("Initializing Engine...")
    t_cfg = cfg.get("trainer", {})
    engine = Engine(
        accelerator=t_cfg.get("accelerator", "gpu"),
        devices=t_cfg.get("devices", 1),
        max_epochs=t_cfg.get("max_epochs", 1),
        default_root_dir=t_cfg.get("default_root_dir", "./results/patchcore"),
    )

    logger.info("Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    return model, engine

def export_onnx(model, cfg: dict, output_path: Path):
    onnx_cfg = cfg.get("export", {}).get("onnx", {})
    input_shape = onnx_cfg.get("input_shape", [1, 3, 256, 256])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    
    torch.onnx.export(
        model.model,
        dummy_input,
        str(output_path),
        opset_version=onnx_cfg.get("opset_version", 17),
        input_names=["input"],
        output_names=["anomaly_map", "anomaly_score"],
        dynamic_axes={"input": {0: "batch_size"}, "anomaly_map": {0: "batch_size"}, "anomaly_score": {0: "batch_size"}},
    )
    logger.info("ONNX exported to %s", output_path)

def main():
    parser = argparse.ArgumentParser(description="Netie AI Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/patchcore_edge.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, engine = train(cfg)
    
    output_dir = Path(cfg["trainer"]["default_root_dir"]) / "weights"
    onnx_path = output_dir / "model.onnx"
    export_onnx(model, cfg, onnx_path)

if __name__ == "__main__":
    main()
