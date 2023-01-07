"""
Adapted from https://huggingface.co/spaces/stabilityai/stable-diffusion
"""

from tensorflow import keras

import time

import gradio as gr
import keras_cv

from constants import css, examples, img_height, img_width, num_images_to_gen
from share_btn import community_icon_html, loading_icon_html, share_js

from huggingface_hub import from_pretrained_keras

PLACEHOLDER_TOKEN="<my-funny-cat-token>"

MODEL_CKPT = "$MODEL_REPO_ID@$MODEL_VERSION"
MODEL = from_pretrained_keras(MODEL_CKPT)

model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height, jit_compile=True
)
model._text_encoder = MODEL
model._text_encoder.compile(jit_compile=True)

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

demoInterface = gr.Interface(
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
    examples=[f"an oil painting of {PLACEHOLDER_TOKEN}", 8], [f"gandalf the gray as a {PLACEHOLDER_TOKEN}", 8]],
    allow_flagging=False,
)

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Your own Stable Diffusion on Google Cloud Platform
    """)
    
    with gr.Row():
        gcp_project_id = gr.Textbox(
            label="GCP project ID",
        )
        gcp_region = gr.Dropdown(
            ["us-central1", "asia‑east1", "asia-northeast1"],
            value="us-central1",
            interactive=True,
            label="GCP Region"
        )

    gr.Markdown(
    """
    Configurations on scalability
    """)        
    with gr.Row():
        min_nodes = gr.Slider(
            label="minimum number of nodes",
            minimum=1,
            maximum=10)
        
        max_nodes = gr.Slider(
            label="maximum number of nodes",
            minimum=1,
            maximum=10)
    
    btn = gr.Button(value="Ready to Deploy!")
    # btn.click(mirror, inputs=[im], outputs=[im_2])    

with gr.Blocks() as demo2:
    gr.Markdown(
    """
    # Your own Stable Diffusion on Hugging Face 🤗 Endpoint
    """)    

gr.TabbedInterface(
    [demoInterface, demo, demo2], ["Try-out", "🚀 Deploy on GCP", " Deploy on 🤗 Endpoint"]
).launch(enable_queue=True)