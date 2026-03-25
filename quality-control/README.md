# Netie AI: Edge-Native Quality Control Pipeline

An open-source, air-gapped machine vision pipeline designed for sub-millisecond anomaly detection in industrial manufacturing. This repository provides a complete MLOps stack for training, compiling, and serving deep learning models on resource-constrained Edge IPCs.

## 1. Project Purpose & Problem Statement
**Industry:** Precision CNC manufacturing (Paradigm Metal Industries)

**Problem:** Manual visual inspection of CNC milled surfaces is slow, inconsistent, and unscalable. Human fatigue often leads to missed defects, and there is no data-driven traceability for trend analysis.

**Solution:** A production-grade edge AI system that:
- Ingests a live RTSP camera feed (global shutter + darkfield lighting).
- Runs PatchCore anomaly detection (Anomalib) at 30+ FPS on NVIDIA hardware.
- Generates pixel-level anomaly heatmaps overlaid on the raw frame.
- Automatically flags defective parts and routes borderline cases for human review.
- **Predictive Maintenance:** Monitor tool wear and process health by analyzing trend shifts in anomaly distributions and data drift.
- **100% Air-Gapped:** Designed to run without any cloud dependency for maximum data sovereignty and security.

## 2. Technology Stack
- **AI Model:** Anomalib PatchCore with WideResNet-50-2 backbone.
- **Inference Runtime:** NVIDIA TensorRT (FP16) for deterministic, ultra-low latency.
- **API Service:** FastAPI wrapper for inference, metrics, and snapshots.
- **Active Learning:** MinIO (Object Storage) + Label Studio (Annotation).
- **Orchestration:** Apache Airflow (Retraining) + K3s (Kubernetes at the Edge).
- **Observability:** Prometheus + Grafana for real-time monitoring and drift detection.

## 3. Architecture Overview
This system utilizes PatchCore to identify surface anomalies without requiring massive datasets of defective samples. The pipeline is optimized for NVIDIA hardware (validated on RTX 4000 series GPUs with 12 GB VRAM).

1. **Model Generation:** Anomalib-driven training loop exporting to ONNX (Opset 17).
2. **Silicon Optimization:** TensorRT compilation yielding FP16 engines for deterministic, ultra-low latency inference.
3. **Inference Serving:** FastAPI wrapper utilizing `pycuda` for direct host-to-device memory allocation.
4. **Active Learning:** Automated borderline-frame detection triggering asynchronous MinIO uploads and Label Studio task generation.
5. **Observability:** Prometheus metrics and Grafana dashboards tracking GPU utilization, inference latency, and concept drift.
6. **Retraining:** Apache Airflow DAGs to merge annotated datasets and trigger automated TensorRT recompilation.

## 4. Quick Start (Docker Compose)
Ensure your host machine has the NVIDIA Container Toolkit installed.

1. Copy `.env.example` to `.env` and configure your RTSP endpoints.
2. Spin up the observability and active learning stack:
   ```bash
   docker compose -f deployments/docker-compose.observability.yml up -d
   ```
3. Build and launch the TensorRT inference service:
   ```bash
   docker build -t netie-ai/inference -f deployments/Dockerfile.tensorrt .
   docker run --gpus all --env-file .env -p 8000:8000 netie-ai/inference
   ```

## 5. License
Distributed under the Apache 2.0 License. See `LICENSE` for more information.
