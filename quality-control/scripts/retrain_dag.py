"""
Netie AI — Active Learning Retraining Pipeline
==============================================
DAG: netie_active_learning_pipeline
"""

from __future__ import annotations
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

logger = logging.getLogger("netie_ai.retrain_dag")

default_args = {
    "owner": "netie-ai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://192.168.1.50:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN", "")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "192.168.1.50:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "your_generic_access_key")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "your_generic_secret_key")
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT_PATH", "/opt/netie-ai/scripts/train_and_export.py")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/opt/netie-ai/configs/patchcore_edge.yaml")
DATASET_DIR = os.getenv("DATASET_DIR", "/opt/netie-ai/data/sample_dataset")
STAGING_DIR = "/tmp/netie_retrain_staging"
BUCKET_NAME = "active-learning-buffer"

def pull_annotations(**context):
    import requests
    from minio import Minio

    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}", "Content-Type": "application/json"}
    # Simplified logic for example: fetch from project 1
    resp = requests.get(f"{LABEL_STUDIO_URL}/api/tasks?project=1", headers=headers, timeout=30)
    resp.raise_for_status()
    tasks = resp.json().get("tasks", [])

    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

    downloaded = []
    for task in tasks:
        if task.get("annotations"):
            object_name = task.get("data", {}).get("object_name")
            if object_name:
                local_path = staging / Path(object_name).name
                minio_client.fget_object(BUCKET_NAME, object_name, str(local_path))
                downloaded.append(str(local_path))

    context["ti"].xcom_push(key="downloaded_images", value=downloaded)

def merge_dataset(**context):
    downloaded = context["ti"].xcom_pull(task_ids="pull_annotations", key="downloaded_images")
    if not downloaded: return
    train_dir = Path(DATASET_DIR) / "good" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for img_path in downloaded:
        shutil.copy2(img_path, train_dir / Path(img_path).name)
    shutil.rmtree(STAGING_DIR)

with DAG(
    dag_id="netie_active_learning_pipeline",
    default_args=default_args,
    description="Automated retraining triggered by human-in-the-loop annotations",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["netie-ai", "mlops"],
) as dag:
    t_pull = PythonOperator(task_id="pull_annotations", python_callable=pull_annotations)
    t_merge = PythonOperator(task_id="merge_dataset", python_callable=merge_dataset)
    t_retrain = BashOperator(task_id="retrain_model", bash_command=f"python {TRAIN_SCRIPT} --config {CONFIG_PATH}")
    
    t_pull >> t_merge >> t_retrain
