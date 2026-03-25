# Netie AI: Architecture & Pipeline Deep Dive

This document provides a technical technical deep dive into the MLOps lifecycle, hardware utilization, and the "predictive maintenance" nature of the Netie AI stack.

## 1. Predictive Maintenance Logic
### What data is ingested?
The system ingests high-resolution frames from industrial RTSP cameras (typically global shutter to minimize motion blur on fast CNC lines). Initial training requires only "Golden Samples" (defect-free surfaces).

### What is the decision based on?
The AI decision is based on the **distance between feature embeddings**. PatchCore extracts deep features from layers 2 and 3 of a WideResNet-50-2 backbone. It uses 10% coreset subsampling to create a compact memory bank of "Normal" features. During inference, any frame whose features fall outside the k=9 nearest neighbor distance in the memory bank is flagged as an anomaly.

### How does it predict maintenance needs?
While primarily a Quality Control (QC) tool, Netie AI serves as a proxy for machine health:
- **Tool Wear Detection:** Gradual shifts in the average anomaly score (even if below the defect threshold) often correlate with tool wear or coolant degradation.
- **Data Drift:** The systems tracks anomaly score distributions over time. A sudden surge in "borderline" frames (scores between 0.35–0.65) triggers a "concept drift" alert in Grafana, signaling that the CNC process has shifted and requires inspection or model retraining.

## 2. Training Pipeline Deep Dive
**Golden CNC Images** → **Anomalib PatchCore** (1-epoch, single-pass)
→ **ONNX Export** (Opset 17, dynamic batch)
→ **TensorRT Engine Build** (FP16 / INT8)

## 3. Real-Time Inference Service
The FastAPI service runs a background thread that continuously grabs RTSP frames.
- **Preprocess:** Resize (256x256), Normalize, HWC→CHW.
- **Inference:** TensorRT Engine utilizing asynchronous CUDA streams for <33ms latency.
- **Overlay:** JET colormap heatmap (α=0.4) for visual verification in dashboards.

## 4. Active Learning Loop (HITL)
The "Active Learning" module handles ambiguous cases:
1. **Borderline Score (0.35-0.65):** Detected at the edge.
2. **Rate-Limit:** 5s cooldown to prevent flooding during transient process issues.
3. **Upload:** Frame saved to MinIO (`active-learning-buffer` bucket).
4. **Task Push:** Label Studio receives a task with a presigned URL for human review.
5. **Retrain:** Apache Airflow DAG (`netie_active_learning_pipeline`) pulls annotations and triggers periodic model updates.

## 5. Monitoring Dashboard (Grafana)
The Grafana dashboard enables technical oversight of the factory floor:
- **Anomaly Score Gauge:** Real-time confidence.
- **Inference FPS:** Targeting ≥25 FPS for smooth line tracking.
- **Camera/Engine Health:** Binary "Online/Offline" status.
- **Inference Latency Histogram:** Tracking deterministic performance.
- **Score Rate of Change:** Used for proactive drift detection.
