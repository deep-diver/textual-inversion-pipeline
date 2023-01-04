# Textual Inversion Pipeline for Stable Diffusion

This repository demonstrates how to gain one's own Stable Diffusion by teaching new concepts of your own images. It uses Stable Diffusion from KerasCV(`==0.4.0`), and the core implementation details of textual inversion is borrowed from [this keras official example](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)

## Overview

![](https://iili.io/HATA65F.png)

The flow of the textual inversion pipeline works as below:

1. (optional) create a new branch 
2. run GitHub Action that triggers fine-tuning process of the embedding layers of `text encoder`
3. fine-tuning process constructed with TFX pipeline runs on Vertex AI pipeline
4. at the end of Vertex AI pipeline, the new `text encoder` with the trained embedding layers is pushed to Hugging Face Model
5. also, the fully working prototype Gradio application is published to Hugging Face Space
6. when you think the result is good enough, click "deploy" button on the Gradio application 
7. then, Stable Diffusion equipped with the new `text encoder` will be deployed to the target platform of your choice
  - Hugging Face Inference Endpoint on AWS and Azure, FastAPI and TF Serving on Google Cloud Platform(GKE)
