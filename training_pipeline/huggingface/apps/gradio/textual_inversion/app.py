"""
Adapted from https://huggingface.co/spaces/stabilityai/stable-diffusion
"""

from tensorflow import keras

import time
import json
import requests

import gradio as gr
import keras_cv

from constants import css, img_height, img_width, num_images_to_gen
from share_btn import community_icon_html, loading_icon_html, share_js

from huggingface_hub import from_pretrained_keras

PLACEHOLDER_TOKEN="$PLACEHOLDER_TOKEN"

MODEL_CKPT = "$MODEL_REPO_ID@$MODEL_VERSION"
MODEL = from_pretrained_keras(MODEL_CKPT)

head_sha = "$SHA"

model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height, jit_compile=True
)
model._text_encoder = MODEL
model._text_encoder.compile(jit_compile=True)
model.tokenizer.add_tokens(PLACEHOLDER_TOKEN)

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
    examples=[
        [f"an oil painting of {PLACEHOLDER_TOKEN}", 8], 
        [f"A mysterious {PLACEHOLDER_TOKEN} approaches the great pyramids of egypt.", 8]],
    allow_flagging=False,
)


def avaliable_providers():
    providers = []
    
    headers = {
        "Content-Type": "application/json",
    }
    endpoint_url = "https://api.endpoints.huggingface.cloud/provider"
    response = requests.get(endpoint_url, headers=headers)

    for provider in response.json()['items']:
        if provider['status'] == 'available':
            providers.append(provider['vendor'])
    
    return providers

def update_regions(provider):
    avalialbe_regions = []
    
    headers = {
        "Content-Type": "application/json",
    }
    endpoint_url = f"https://api.endpoints.huggingface.cloud/provider/{provider}/region"
    response = requests.get(endpoint_url, headers=headers)

    for region in response.json()['items']:
        if region['status'] == 'available':
            avalialbe_regions.append(f"{region['region']}/{region['label']}")

    return gr.Dropdown.update(
        choices=avalialbe_regions,
        value=avalialbe_regions[0] if len(avalialbe_regions) > 0 else None
    )

def update_compute_options(provider, region):
    region = region.split("/")[0]
    avalialbe_compute_options = []

    headers = {
        "Content-Type": "application/json",
    }
    endpoint_url = f"https://api.endpoints.huggingface.cloud/provider/{provider}/region/{region}/compute"
    print(endpoint_url)
    response = requests.get(endpoint_url, headers=headers)

    for compute in response.json()['items']:
        if compute['status'] == 'available':
            accelerator = compute['accelerator']
            numAccelerators = compute['numAccelerators']
            memoryGb = compute['memoryGb'].replace("Gi", "GB")
            architecture = compute['architecture']
            
            type = f"{numAccelerators}vCPU {memoryGb} · {architecture}" if accelerator == "cpu" else f"{numAccelerators}x {architecture}"
            
            avalialbe_compute_options.append(
                f"{compute['accelerator'].upper()} [{compute['instanceSize']}] · {type}"
            )

    return gr.Dropdown.update(
        choices=avalialbe_compute_options,
        value=avalialbe_compute_options[0] if len(avalialbe_compute_options) > 0 else None
    )

def submit(
    hf_account_input,
    hf_token_input,
    endpoint_name_input,
    provider_selector,
    region_selector,
    repository_selector,
    task_selector,
    framework_selector,
    compute_selector,
    min_node_selector, 
    max_node_selector, 
    security_selector    
):
    compute_resources = compute_selector.split("·")
    accelerator = compute_resources[0][:3].strip()

    size_l_index = compute_resources[0].index("[") - 1
    size_r_index = compute_resources[0].index("]")
    size = compute_resources[0][size_l_index : size_r_index].strip()

    type = compute_resources[-1].strip()
    
    payload = {
      "accountId": hf_account_input.strip(),
      "compute": {
        "accelerator": accelerator.lower(),
        "instanceSize": size[1:],
        "instanceType": type,
        "scaling": {
          "maxReplica": int(max_node_selector),
          "minReplica": int(min_node_selector)
        }
      },
      "model": {
        "framework": framework_selector.lower(),
        "image": {
          "huggingface": {}
        },
        "repository": repository_selector.lower(),
        "revision": head_sha,
        "task": task_selector.lower()
      },
      "name": endpoint_name_input.strip(),
      "provider": {
        "region": region_selector.split("/")[0].lower(),
        "vendor": provider_selector.lower()
      },
      "type": security_selector.lower()
    }
    
    print(payload)

    payload = json.dumps(payload)
    print(payload)

    headers = {
        "Authorization": f"Bearer {hf_token_input.strip()}",
        "Content-Type": "application/json",
    }
    endpoint_url = f"https://api.endpoints.huggingface.cloud/endpoint"
    print(endpoint_url)

    response = requests.post(endpoint_url, headers=headers, data=payload)

    if response.status_code == 400:
        return f"{response.text}. Malformed data in {payload}"
    elif response.status_code == 401:
        return "Invalid token"
    elif response.status_code == 409:
        return f"Endpoint {endpoint_name_input} already exists"
    elif response.status_code == 202:
        return f"Endpoint {endpoint_name_input} created successfully on {provider_selector.lower()} using {repository_selector.lower()}@{head_sha}.\nPlease check out the progress at https://ui.endpoints.huggingface.co/endpoints."
    else:
        return f"something went wrong {response.status_code} = {response.text}"

with gr.Blocks() as hf_endpoint:
    providers = avaliable_providers()

    gr.Markdown(
    """
    ## Deploy Stable Diffusion on 🤗 Endpoint
    ---
    """)
    
    gr.Markdown("""
    #### Your 🤗 Account ID(Name)
    """)
    hf_account_input = gr.Textbox(
        show_label=False,
    )

    gr.Markdown("""
    #### Your 🤗 Access Token
    """)
    hf_token_input = gr.Textbox(
        show_label=False,
        type="password"
    )

    gr.Markdown("""    
    #### Decide the Endpoint name
    """)
    endpoint_name_input = gr.Textbox(
        show_label=False
    )    

    with gr.Row():
        gr.Markdown("""    
        #### Cloud Provider
        """)

        gr.Markdown("""    
        #### Cloud Region
        """)       
    
    with gr.Row():
        provider_selector = gr.Dropdown(
            choices=providers,
            interactive=True,
            show_label=False,
        )
        
        region_selector = gr.Dropdown(
            [],
            value="",
            interactive=True,
            show_label=False,
        )
        
        provider_selector.change(update_regions, inputs=provider_selector, outputs=region_selector)

    with gr.Row():
        gr.Markdown("""    
        #### Target Model
        """)

        gr.Markdown("""    
        #### Target Model Version(branch)
        """)       
    
    with gr.Row():
        repository_selector = gr.Textbox(
            value="$MODEL_REPO_ID",
            interactive=False,
            show_label=False,
        )

        revision_selector = gr.Textbox(
            value=f"$MODEL_VERSION/{head_sha[:7]}",
            interactive=False,
            show_label=False,
        )        

    with gr.Row():
        gr.Markdown("""    
        #### Task
        """)

        gr.Markdown("""    
        #### Framework
        """)      
    
    with gr.Row():
        task_selector = gr.Textbox(
            value="Custom",
            interactive=False,
            show_label=False,
        )

        framework_selector = gr.Textbox(
            value="TensorFlow",
            interactive=False,
            show_label=False,
        )

    gr.Markdown("""
    
    #### Select Compute Instance Type
    """)    
    compute_selector = gr.Dropdown(
        [],
        value="",
        interactive=True,
        show_label=False,
    )
    region_selector.change(update_compute_options, inputs=[provider_selector, region_selector], outputs=compute_selector)

    with gr.Row():
        gr.Markdown("""    
        #### Min Number of Nodes
        """)

        gr.Markdown("""    
        #### Max Number of Nodes
        """)

        gr.Markdown("""    
        #### Security Level
        """)        
    
    with gr.Row():
        min_node_selector = gr.Number(
            value=1,
            interactive=True,
            show_label=False,
        )

        max_node_selector = gr.Number(
            value=1,
            interactive=True,
            show_label=False,
        )

        security_selector = gr.Radio(
            choices=["Protected", "Public", "Private"],
            value="Public",
            interactive=True,
            show_label=False,
        )
    
    submit_button = gr.Button(
        value="Submit",
    )

    status_txt = gr.Textbox(
        value="any status update will be displayed here",
        interactive=False
    )

    submit_button.click(
        submit, 
        inputs=[
            hf_account_input,
            hf_token_input,
            endpoint_name_input,
            provider_selector,
            region_selector,
            repository_selector,
            task_selector,
            framework_selector,
            compute_selector,
            min_node_selector, 
            max_node_selector, 
            security_selector],
        outputs=status_txt)

gr.TabbedInterface(
    [demoInterface, hf_endpoint], ["Playground", " Deploy on 🤗 Endpoint"]
).launch(enable_queue=True)
