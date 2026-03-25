"""
Netie AI — Real-Time Edge Inference Service
==========================================
FastAPI application for industrial quality control.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .trt_wrapper import TRTInferenceEngine
from .telemetry import (
    ANOMALY_SCORE,
    INFERENCE_LATENCY,
    FRAMES_PROCESSED,
    INFERENCE_FPS,
    DEFECTS_DETECTED,
    CAMERA_CONNECTED,
    ENGINE_LOADED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("netie_ai.service")

# ── Configuration via env vars ────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_STREAM_URL", "rtsp://admin:password@192.168.1.100/ch1/main")
ENGINE_PATH = os.getenv("TRT_ENGINE_PATH", "/opt/models/patchcore_fp16.engine")
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
LOW_CONF_MIN = float(os.getenv("ANOMALY_THRESHOLD_MIN", "0.4"))
LOW_CONF_MAX = float(os.getenv("ANOMALY_THRESHOLD_MAX", "0.6"))

# ── Shared state ──────────────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
latest_overlay: np.ndarray | None = None
latest_score: float = 0.0
pipeline_running = threading.Event()
lock = threading.Lock()


def get_active_learner():
    try:
        from .active_learner import ActiveLearner
        return ActiveLearner(
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "192.168.1.50:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "your_generic_access_key"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "your_generic_secret_key"),
            label_studio_url=os.getenv("LABEL_STUDIO_URL", "http://192.168.1.50:8080"),
            label_studio_token=os.getenv("LABEL_STUDIO_API_TOKEN", ""),
            low_conf_min=LOW_CONF_MIN,
            low_conf_max=LOW_CONF_MAX,
        )
    except Exception as e:
        logger.warning("Active learning module unavailable: %s", e)
        return None


def inference_loop(engine: TRTInferenceEngine, active_learner):
    global latest_frame, latest_overlay, latest_score

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Cannot open RTSP stream: %s", RTSP_URL)
        CAMERA_CONNECTED.set(0)
        return

    CAMERA_CONNECTED.set(1)
    fps_window = []
    pipeline_running.set()

    try:
        while pipeline_running.is_set():
            ret, frame = cap.read()
            if not ret:
                CAMERA_CONNECTED.set(0)
                cap.release()
                time.sleep(2.0)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    CAMERA_CONNECTED.set(1)
                continue

            t0 = time.perf_counter()
            score, anomaly_map = engine.infer(frame)
            latency = time.perf_counter() - t0
            overlay = engine.overlay_heatmap(frame, anomaly_map)

            with lock:
                latest_frame = frame.copy()
                latest_overlay = overlay.copy()
                latest_score = score

            ANOMALY_SCORE.set(score)
            INFERENCE_LATENCY.observe(latency)
            FRAMES_PROCESSED.inc()

            fps_window.append(latency)
            if len(fps_window) > 30:
                fps_window.pop(0)
            avg_latency = sum(fps_window) / len(fps_window)
            INFERENCE_FPS.set(1.0 / avg_latency if avg_latency > 0 else 0)

            if score > ANOMALY_THRESHOLD:
                DEFECTS_DETECTED.inc()

            if active_learner is not None:
                active_learner.process_frame(frame, score)
    finally:
        cap.release()
        CAMERA_CONNECTED.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TensorRT engine: %s", ENGINE_PATH)
    engine = TRTInferenceEngine(ENGINE_PATH)
    ENGINE_LOADED.set(1)
    active_learner = get_active_learner()

    thread = threading.Thread(
        target=inference_loop,
        args=(engine, active_learner),
        daemon=True,
    )
    thread.start()
    yield
    pipeline_running.clear()
    thread.join(timeout=5)
    ENGINE_LOADED.set(0)


app = FastAPI(
    title="Netie AI Edge QC Pipeline",
    description="Sub-millisecond anomaly detection for Industrial IPCs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "engine_loaded": bool(ENGINE_LOADED._value._value), "camera_connected": bool(CAMERA_CONNECTED._value._value)}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/snapshot")
async def snapshot():
    with lock:
        frame = latest_overlay
    if frame is None:
        return Response(content="No frame", status_code=503)
    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
