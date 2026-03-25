"""
Netie AI — Prometheus Metrics
============================
Defines all Prometheus metrics exposed by the inference service.
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
)

# ── Application info ──────────────────────────────────────────────────────
APP_INFO = Info(
    "netie_edge_inference",
    "Netie AI Edge-Native Quality Control Pipeline metadata",
)
APP_INFO.info(
    {
        "version": "1.0.0",
        "model": "PatchCore",
        "precision": "FP16",
    }
)

# ── Core inference metrics ────────────────────────────────────────────────
ANOMALY_SCORE = Gauge(
    "netie_anomaly_score_gauge",
    "Real-time anomaly confidence (higher = more anomalous)",
)

INFERENCE_LATENCY = Histogram(
    "netie_inference_latency_histogram",
    "End-to-end processing time per frame in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

FRAMES_PROCESSED = Counter(
    "netie_frames_processed_total",
    "Total number of frames processed by the inference engine",
)

INFERENCE_FPS = Gauge(
    "netie_inference_fps_gauge",
    "Rolling frames-per-second calculation",
)

# ── Active learning metrics ───────────────────────────────────────────────
BORDERLINE_FRAMES = Counter(
    "netie_borderline_frames_total",
    "Total frames flagged as borderline for HITL review",
)

DEFECTS_DETECTED = Counter(
    "netie_defects_detected_total",
    "Total frames classified as defective (score > threshold)",
)

# ── System health ─────────────────────────────────────────────────────────
CAMERA_CONNECTED = Gauge(
    "netie_camera_connected",
    "RTSP camera connection status (1=CONNECTED, 0=DISCONNECTED)",
)

ENGINE_LOADED = Gauge(
    "netie_engine_loaded",
    "TensorRT engine status (1=READY, 0=NOT_LOADED)",
)
