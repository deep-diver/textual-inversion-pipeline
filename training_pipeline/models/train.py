from typing import List

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from keras_cv.models import StableDiffusion
from keras_cv.models.stable_diffusion import NoiseScheduler
from keras_cv import layers as cv_layers

from .utils import traverse_layers
from .prepare_text_encoder import prepare_text_encoder
from .prepare_dataset import prepare_text_dataset, prepare_image_dataset
from .finetuner import StableDiffusionFineTuner

from tfx.components.trainer.fn_args_utils import FnArgs

def set_trainable_parameters(stable_diffusion):
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

def get_imenc_and_scheduler(stable_diffusion):
    image_encoder = keras.Model(stable_diffusion.image_encoder.input, stable_diffusion.image_encoder.layers[-2].output)
    image_encoder.get_layer(index=0)._name = "images"

    noise_scheduler = NoiseScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", train_timesteps=1000
    )

    return image_encoder, noise_scheduler

def get_optimizer(train_ds, epochs):
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4, 
        decay_steps=train_ds.cardinality()*EPOCHS)

    optimizer = keras.optimizers.Adam(
        weight_decay=0.004, 
        learning_rate=learning_rate, 
        epsilon=1e-8, 
        global_clipnorm=10
    )
    return optimizer

def run_fn(fn_args: FnArgs):
    hyperparameters = fn_args.custom_config["hyperparameters"]
    epochs = hyperparameters['epoch']
    initialized_target_token = hyperparameters['initialized_target_token']
    print(f"epochs: {epochs}")
    print(f"initialized_target_token: {initialized_target_token}")

    stable_diffusion = StableDiffusion()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    image_dataset = prepare_image_dataset(fn_args.train_files, fn_args.data_accessor, tf_transform_output)
    text_dataset = prepare_text_dataset(stable_diffusion)

    train_ds = tf.data.Dataset.zip((image_dataset, text_dataset))
    train_ds = train_ds.repeat(10).shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)

    _ = prepare_text_encoder(stable_diffusion, initialized_target_token=initialized_target_token)
    set_trainable_parameters(stable_diffusion)

    img_encoder, scheduler = get_imenc_and_scheduler(stable_diffusion)
    trainer = StableDiffusionFineTuner(stable_diffusion, scheduler, img_encoder, name="trainer")

    trainer.compile(
        optimizer=get_optimizer(train_ds, epochs),
        loss=keras.losses.MeanSquaredError(reduction="none"),
    )
    trainer.fit(
        train_ds,
        epochs=epochs,
    )

    stable_diffusion.text_encoder.save(
        fn_args.serving_model_dir,
        save_format="tf"
    )    
