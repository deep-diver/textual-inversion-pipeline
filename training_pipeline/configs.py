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

TRAINING_EPOCH = 1
INITIALIZED_TARGET_TOKEN = "cat"
PLACEHOLDER_TOKEN = "<my-funny-cat-token>"

GRADIO_APP_PATH = "apps.gradio.textual_inversion"

TRAINING_CUSTOM_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_training_const.TRAINING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "a2-highgpu-1g",
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
    "hyperparameters": {
        "epoch": TRAINING_EPOCH,
        "initialized_target_token": INITIALIZED_TARGET_TOKEN,
        "placeholder_token": PLACEHOLDER_TOKEN
    }
}

HF_PUSHER_ARGS = {
    "username": "chansung",
    "access_token": "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",
    "repo_name": PIPELINE_NAME,
    "space_config": {
        "app_path": GRADIO_APP_PATH,
        "additional_replacements": {
            "$PLACEHOLDER_TOKEN": PLACEHOLDER_TOKEN
        }
    },
}