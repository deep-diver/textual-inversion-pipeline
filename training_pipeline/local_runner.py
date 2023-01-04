import os
from absl import logging

from tfx import v1 as tfx
from tfx.orchestration.data_types import RuntimeParameter
from pipeline import configs
from pipeline import local_pipeline

OUTPUT_DIR = "."

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(
    OUTPUT_DIR, "tfx_metadata", configs.PIPELINE_NAME, "metadata.db"
)

def run():
    """Define a pipeline."""

    tfx.orchestration.LocalDagRunner().run(
        local_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            modules={
                "training_fn": configs.TRAINING_FN,
            },
            train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            hf_pusher_args=configs.HF_PUSHER_ARGS,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                METADATA_PATH
            ),
        )
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()