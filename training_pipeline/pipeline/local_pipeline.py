from typing import Any, Dict, List, Optional, Text

from tfx import v1 as tfx
import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2

import absl
from tfx.components import ImportExampleGen, Trainer, Transform
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel

from pipeline.components.HFPusher.component import HFPusher

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    schema_path: Text,
    modules: Dict[Text, Text],
    training_custom_args: Optional[Dict[Text, Text]] = None,
    hf_pusher_args: Optional[Dict[Text, Any]],
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
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

    trainer = Trainer(
        run_fn=modules["training_fn"],
        transformed_examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        custom_config=training_custom_args,
    )
    components.append(trainer)

    hf_pusher_args["model"] = trainer.outputs["model"]
    hf_pusher = HFPusher(**hf_pusher_args)
    components.append(hf_pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )