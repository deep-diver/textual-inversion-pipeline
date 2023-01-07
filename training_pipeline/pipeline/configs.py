import os

import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const

PIPELINE_NAME = "$PIPELINE_NAME"

try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"
except ImportError:
    GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"

GOOGLE_CLOUD_REGION = "us-central1"

DATA_PATH = "tfrecords"
SCHEMA_PATH = "pipeline/schema.pbtxt" # GCS path is also allowed

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + "-complete-mlops"
PIPELINE_IMAGE = f"gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}"

OUTPUT_DIR = os.path.join("gs://", GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", PIPELINE_NAME)

TRAINING_FN = "models.train.run_fn"
PREPROCESSING_FN = "dataprocessings.preprocessing.preprocessing_fn"

GRADIO_APP_PATH = "apps.gradio.img_classifier"
MODEL_HUB_REPO_PLACEHOLDER = "$MODEL_REPO_ID"
MODEL_HUB_URL_PLACEHOLDER = "$MODEL_REPO_URL"
MODEL_VERSION_PLACEHOLDER = "$MODEL_VERSION"

TRAIN_NUM_STEPS = 160
EVAL_NUM_STEPS = 4

GCP_AI_PLATFORM_TRAINING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_training_const.TRAINING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_A100",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": PIPELINE_IMAGE,
                },
            }
        ],
    },
    "use_gpu": True,
}

HF_PUSHER_ARGS = {
    "username": "chansung",
    "access_token": "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",
    "repo_name": PIPELINE_NAME,
    "space_config": {
        "app_path": "apps.gradio.textual_inversion",
    },
}