from typing import Any, Dict, List, Optional, Text

import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2
from tfx import v1 as tfx
from tfx.components import (
    ImportExampleGen,
    Transform,
)

from tfx.extensions.google_cloud_ai_platform.trainer.component import (
    Trainer as VertexTrainer,
)
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from pipeline.components.HFPusher.component import HFPusher


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    schema_path: Text,
    modules: Dict[Text, Text],
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    hf_pusher_args: Optional[Dict[Text, Any]] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="*.tfrecord")
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
    components.append(example_gen)

    schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
    components.append(schema_gen)

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn=modules["preprocessing_fn"],
    )
    components.append(transform)    

    # Training.
    trainer_args = {
        "run_fn": modules["training_fn"],
        "transformed_examples": transform.outputs["transformed_examples"],
        "transform_graph": transform.outputs["transform_graph"],
        "custom_config": ai_platform_training_args,
    }
    trainer = VertexTrainer(**trainer_args)
    components.append(trainer)

    # Push the blesses model to HF hub and deploy a demo app on Hugging Face Spaces.
    hf_pusher_args["model"] = trainer.outputs["model"]
    hf_pusher = HFPusher(**hf_pusher_args)
    components.append(hf_pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
    )