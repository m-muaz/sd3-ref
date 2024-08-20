import os
from glob import glob
import torch, fire, math
from diffusers import StableDiffusion3Pipeline



def main(model_path=None):
    if model_path is not None:
        # Make the path absolute and make sure it exists
        model_path = os.path.abspath(model_path)
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    else:
        # Otherwise use huggingface model path 
        model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(
    "../stable-diffusion-3-medium", torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    
    
if __name__ == "__main__":
    fire.Fire(main)