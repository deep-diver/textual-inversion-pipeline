import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv import layers as cv_layers

import utils

def prepare_images(folder, extension):
    files = glob.blog(f"{folder}/*.{extension")
    files = [tf.keras.utils.get_file(origin=f) for f in files]

    resize = keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    images = [keras.utils.load_img(img) for img in files]
    images = [keras.utils.img_to_array(img) for img in images]
    images = np.array([resize(img) for img in images])
    images = images / 127.5 - 1

    return images

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
    object_prompts = prepare_prompts(placeholder_token)

    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in object_prompts]

    stable_diffusion.tokenizer.add_tokens(placeholder_token)
    # Create new embeddings based on the old ones.

    # Replace with style_prompts if you'd like to finetune on a style
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in object_prompts]
    return embeddings

def pad_embedding(stable_diffusion, embedding):
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text]
        * (utils.MAX_PROMPT_LENGTH - len(embedding))
    )

def prepare_image_dataset(folder, extension="jpeg"):
    images = prepare_images(folder, extension)
    
    image_dataset = tf.data.Dataset.from_tensor_slices(images)
    image_dataset = image_dataset.shuffle(100)
    image_dataset = image_dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.map(
        cv_layers.RandomFlip(mode="horizontal"), num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.repeat()

    return image_dataset

def prepare_text_dataset(stable_diffusion, placeholder_token="<benny-the-aussie>"):
    embeddings = prepare_embeddings(stable_diffusion, placeholder_token)

    embeddings = [np.array(pad_embedding(stable_diffusion, embedding)) for embedding in embeddings]
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)

    return text_dataset