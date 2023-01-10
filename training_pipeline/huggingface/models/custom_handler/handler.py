from typing import Dict, Any

import sys
import base64
import logging

import tensorflow as tf
import keras_cv

class EndpointHandler():
    def __init__(self, path=""):
        self.sd = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
        self.sd.text_to_image("test prompt", batch_size=1)

        sd._text_encoder = tf.saved_model.load(path)
        sd._text_encoder.compile(jit_compile=True)
        sd.tokenizer.add_tokens("$PLACEHOLDER_TOKEN")
    
    def __call__(self, data: Dict[str, Any]) -> str:
        prompt = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        images = self.sd.text_to_image(prompt, batch_size=batch_size)
        return base64.b64encode(images.tobytes()).decode()