from typing import Any, Dict, List, Optional, Text

from tfx import v1 as tfx
import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2

import absl
from tfx.components import Trainer
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel

from pipeline.components.HFPusher.component import HFPusher

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    modules: Dict[Text, Text],
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    hf_pusher_args: Optional[Dict[Text, Any]],
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    components = []

    trainer = Trainer(
        run_fn=modules["training_fn"],
        transformed_examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=train_args,
        eval_args=eval_args,
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