from typing import List

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

from .common import IMAGE_KEY, LABEL_KEY, NUM_LABELS
from .hyperparams import EPOCHS, EVAL_BATCH_SIZE, TRAIN_BATCH_SIZE
from .signatures import (
    model_exporter,
    tf_examples_serving_signature,
    transform_features_signature,
)
from .unet import build_model
from .utils import transformed_name

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE,
    )
    model = build_model(
        transformed_name(IMAGE_KEY), transformed_name(LABEL_KEY), NUM_LABELS
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures={
            "serving_default": model_exporter(model),
            "transform_features": transform_features_signature(
                model, tf_transform_output
            ),
            "from_examples": tf_examples_serving_signature(model, tf_transform_output),
        },
    )