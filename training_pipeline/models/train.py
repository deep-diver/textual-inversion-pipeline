from typing import List

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from keras_cv.models import StableDiffusion
from keras_cv.models.stable_diffusion import NoiseScheduler
from keras_cv import layers as cv_layers

from .utils import traverse_layers
from .prepare_text_encoder import prepare_text_encoder
from .prepare_dataset import prepare_text_dataset
from .finetuner import StableDiffusionFineTuner

from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

def _input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput
) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=1, shuffle=False
        ),        
        tf_transform_output.transformed_metadata.schema,
    )

    dataset = dataset.shuffle(50, reshuffle_each_iteration=True)
    dataset = dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        cv_layers.RandomFlip(mode="horizontal"), num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.repeat()

    return dataset

def run_fn(fn_args: FnArgs):
    EPOCHS=30

    stable_diffusion = StableDiffusion()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    image_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
    )
    text_dataset = prepare_text_dataset(stable_diffusion)

    train_ds = tf.data.Dataset.zip((image_dataset, text_dataset))
    train_ds = train_ds.repeat(10).shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)

    _ = prepare_text_encoder(stable_diffusion)

    stable_diffusion.diffusion_model.trainable = False
    stable_diffusion.decoder.trainable = False
    stable_diffusion.text_encoder.trainable = True

    stable_diffusion.text_encoder.layers[2].trainable = True

    for layer in traverse_layers(stable_diffusion.text_encoder):
        if isinstance(layer, keras.layers.Embedding) or "clip_embedding" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    stable_diffusion.text_encoder.layers[2].position_embedding.trainable=False

    training_image_encoder = keras.Model(stable_diffusion.image_encoder.input, stable_diffusion.image_encoder.layers[-2].output)
    training_image_encoder.get_layer(index=0)._name = "images"

    noise_scheduler = NoiseScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", train_timesteps=1000
    )
    trainer = StableDiffusionFineTuner(stable_diffusion, noise_scheduler, training_image_encoder, name="trainer")

    learning_rate = keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=train_ds.cardinality()*EPOCHS)
    optimizer = keras.optimizers.Adam(
        weight_decay=0.004, learning_rate=learning_rate, epsilon=1e-8, global_clipnorm=10
    )

    trainer.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(reduction="none"),
    )

    trainer.fit(
        train_ds,
        epochs=EPOCHS,
    )

    stable_diffusion.text_encoder.save(
        fn_args.serving_model_dir,
        save_format="tf"
    )    
