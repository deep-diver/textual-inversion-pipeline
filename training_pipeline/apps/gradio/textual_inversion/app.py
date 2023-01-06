"""
Adapted from https://huggingface.co/spaces/stabilityai/stable-diffusion
"""

from tensorflow import keras

import time

import gradio as gr
import keras_cv

from constants import css, examples, img_height, img_width, num_images_to_gen
from share_btn import community_icon_html, loading_icon_html, share_js

MODEL_CKPT = "$MODEL_REPO_ID@$MODEL_VERSION"
MODEL = from_pretrained_keras(MODEL_CKPT)

model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height, jit_compile=True
)
model.text_encoder = MODEL
model.text_encoder.compile(jit_compile=True)

# Warm-up the model.
_ = model.text_to_image("Teddy bear", batch_size=num_images_to_gen)

def generate_image_fn(prompt: str, unconditional_guidance_scale: int) -> list:
    start_time = time.time()
    # `images is an `np.ndarray`. So we convert it to a list of ndarrays.
    # Each ndarray represents a generated image.
    # Reference: https://gradio.app/docs/#gallery
    images = model.text_to_image(
        prompt,
        batch_size=num_images_to_gen,
        unconditional_guidance_scale=unconditional_guidance_scale,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
    return [image for image in images]


description = "This Space demonstrates a fine-tuned Stable Diffusion model."
article = "This Space is generated automatically from a TFX pipeline. If you are interested in, please check out the [original repository](https://github.com/deep-diver/textual-inversion-sd)."
gr.Interface(
    generate_image_fn,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            max_lines=1,
#            placeholder="cute Sundar Pichai creature",
        ),
        gr.Slider(value=40, minimum=8, maximum=50, step=1),
    ],
    outputs=gr.Gallery().style(grid=[2], height="auto"),
    title="Generate custom images with finetuned embeddings of Stable Diffusion",
    description=description,
    article=article,
    examples=[["cute Sundar Pichai creature", 8], ["Hello kitty", 8]],
    allow_flagging=False,
).launch(enable_queue=True)