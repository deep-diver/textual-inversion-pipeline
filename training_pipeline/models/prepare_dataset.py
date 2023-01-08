import glob
import mimetypes
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv import layers as cv_layers

import tensorflow_transform as tft
from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs

from .utils import MAX_PROMPT_LENGTH

def prepare_image_dataset(
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

def prepare_prompts(placeholder_token):
    object_prompts = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    object_prompts = [prompt.format(placeholder_token) for prompt in object_prompts]
    return object_prompts

def prepare_embeddings(stable_diffusion, placeholder_token):
    stable_diffusion.tokenizer.add_tokens(placeholder_token)
    
    object_prompts = prepare_prompts(placeholder_token)
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in object_prompts]
    return embeddings

def pad_embedding(stable_diffusion, embedding):
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text]
        * (MAX_PROMPT_LENGTH - len(embedding))
    )

def prepare_text_dataset(stable_diffusion, placeholder_token="<my-funny-cat-token>"):
    embeddings = prepare_embeddings(stable_diffusion, placeholder_token)
    embeddings = [np.array(pad_embedding(stable_diffusion, embedding)) for embedding in embeddings]
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    text_dataset.repeat(5)
    return text_dataset
