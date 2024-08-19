import os
from glob import glob
import torch, fire, math
from diffusers import StableDiffusion3Pipeline



def main():
    pipe = StableDiffusion3Pipeline.from_pretrained(
    "../stable-diffusion-3-medium", torch_dtype=torch.float16
    ).to("cuda")

        image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]