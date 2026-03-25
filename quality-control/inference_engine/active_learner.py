"""
Netie AI — Active Learning Edge Buffer
=======================================
Detects borderline inference frames (anomaly scores in the ambiguous zone)
and routes them to MinIO + Label Studio for Human-in-the-Loop review.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from datetime import datetime, timezone
from threading import Lock

import cv2
import numpy as np
import requests

logger = logging.getLogger("netie_ai.active_learner")

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    raise ImportError("minio package required: pip install minio")


class ActiveLearner:
    """Thread-safe borderline frame handler with MinIO + Label Studio integration.

    Args:
        minio_endpoint:     MinIO host:port (e.g. 'minio:9000')
        minio_access_key:   MinIO access key
        minio_secret_key:   MinIO secret key
        label_studio_url:   Base URL of Label Studio (e.g. 'http://label-studio:8080')
        label_studio_token: API token for Label Studio
        low_conf_min:       Lower bound of borderline zone (default 0.4)
        low_conf_max:       Upper bound of borderline zone (default 0.6)
        bucket_name:        MinIO bucket for borderline images
        ls_project_id:      Label Studio project ID to push tasks to
        cooldown_seconds:   Min seconds between consecutive uploads (rate limit)
    """

    def __init__(
        self,
        minio_endpoint: str = "minio:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        label_studio_url: str = "http://label-studio:8080",
        label_studio_token: str = "",
        low_conf_min: float = 0.4,
        low_conf_max: float = 0.6,
        bucket_name: str = "active-learning-buffer",
        ls_project_id: int = 1,
        cooldown_seconds: float = 5.0,
    ):
        self.low_conf_min = low_conf_min
        self.low_conf_max = low_conf_max
        self.bucket_name = bucket_name
        self.ls_project_id = ls_project_id
        self.label_studio_url = label_studio_url.rstrip("/")
        self.label_studio_token = label_studio_token
        self.cooldown_seconds = cooldown_seconds

        self._lock = Lock()
        self._last_upload_time: float = 0.0
        self._total_buffered: int = 0

        # MinIO client (not using TLS in air-gapped LAN)
        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create the active learning bucket if it doesn't exist."""
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                logger.info("Created MinIO bucket: %s", self.bucket_name)
        except S3Error as e:
            logger.error("MinIO bucket check failed: %s", e)

    def _upload_to_minio(self, frame: np.ndarray, object_name: str) -> str:
        """Encode frame as PNG and upload to MinIO."""
        success, png_bytes = cv2.imencode(".png", frame)
        if not success:
            raise RuntimeError("Failed to encode frame to PNG")

        data = io.BytesIO(png_bytes.tobytes())
        data_length = data.getbuffer().nbytes

        self.minio_client.put_object(
            self.bucket_name,
            object_name,
            data,
            length=data_length,
            content_type="image/png",
        )
        logger.info("Uploaded to MinIO: %s/%s", self.bucket_name, object_name)

        from datetime import timedelta
        presigned_url = self.minio_client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=timedelta(days=7),
        )
        return presigned_url

    def _push_to_label_studio(
        self,
        image_url: str,
        anomaly_score: float,
        object_name: str,
    ) -> bool:
        """Create an annotation task in Label Studio."""
        if not self.label_studio_token:
            logger.warning("Label Studio token not set — skipping task push")
            return False

        headers = {
            "Authorization": f"Token {self.label_studio_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "data": {
                "image": image_url,
                "anomaly_score": anomaly_score,
                "source": "active_learner",
                "object_name": object_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "project": self.ls_project_id,
        }

        try:
            resp = requests.post(
                f"{self.label_studio_url}/api/tasks",
                json=payload,
                headers=headers,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logger.info("Label Studio task created for score=%.3f", anomaly_score)
                return True
            else:
                logger.error("Label Studio error %d: %s", resp.status_code, resp.text[:200])
                return False
        except requests.RequestException as e:
            logger.error("Label Studio request failed: %s", e)
            return False

    def process_frame(self, frame: np.ndarray, anomaly_score: float) -> bool:
        """Process a frame: if borderline, upload to MinIO and push to Label Studio."""
        if not (self.low_conf_min <= anomaly_score <= self.low_conf_max):
            return False

        with self._lock:
            now = time.time()
            if now - self._last_upload_time < self.cooldown_seconds:
                return False
            self._last_upload_time = now
            self._total_buffered += 1

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        object_name = f"borderline/{ts}_{uid}_score{anomaly_score:.3f}.png"

        try:
            presigned_url = self._upload_to_minio(frame, object_name)
            self._push_to_label_studio(presigned_url, anomaly_score, object_name)

            try:
                from .telemetry import BORDERLINE_FRAMES
                BORDERLINE_FRAMES.inc()
            except ImportError:
                pass

            return True
        except Exception as e:
            logger.error("Active learning processing failed: %s", e)
            return False

    @property
    def total_buffered(self) -> int:
        return self._total_buffered
