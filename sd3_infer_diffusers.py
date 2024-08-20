import os
from glob import glob
import torch
import fire
import math  # noqa: F401
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = None
NEG_PROMPT=""
# OR use txt file to load prompts
PROMPT_PATH=None
NEG_PROMPT_PATH=None
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 50
# Random seed
SEED = 42
# Optional init image file path
INIT_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.6
# Output file path
OUTPUT = "./sampled_images"
# DEVICE [cuda, cpu]
DEVICE = "cuda"

# == Load prompts from txt file ==
def load_prompts(prompt_path, start_idx=None, end_idx=None):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    prompts = prompts[start_idx:end_idx]
    return prompts

def main(
    model_path=None,
    prompt=PROMPT,
    prompt_path=PROMPT_PATH,
    negative_prompt=NEG_PROMPT,
    negative_prompt_path=NEG_PROMPT_PATH,
    width=WIDTH,
    height=HEIGHT,
    steps=STEPS,
    cfg_scale=CFG_SCALE,
    seed=SEED,
    output=OUTPUT,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    device=DEVICE,
):
    if model_path is not None:
        # Make the path absolute and make sure it exists
        model_path = os.path.abspath(model_path)
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    else:
        # Otherwise use huggingface model path 
        model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        
    #== Make sure OUTPUT dir exists ==
    os.makedirs(output, exist_ok=True)
    batch_size = 1
    # Load in positive prompts
    if prompt is None:
        if prompt_path is not None:
            prompts = load_prompts(prompt_path)
        else:
            prompts = ["a cat holding a sign that says hello world"]
    else:
        prompts = [prompt]
    
    # Load in negative prompts
    if negative_prompt_path is not None:
        negative_prompts = load_prompts(negative_prompt_path)
    else:
        negative_prompts = [negative_prompt] * len(prompts)
 
    
    assert len(prompts) == len(negative_prompts), "Number of prompts and negative prompts must match"
    
    #== Flow Match Euler Discrete Scheduler ==
    print("Loading Scheduler + Pipeline...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    generator = torch.Generator(device=device).manual_seed(seed)
    pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path, scheduler=scheduler, torch_dtype=torch.float16
    ).to(device)
    print("Done!")
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        prompt = prompts[i:i+batch_size] # FIX: handle case where batch_size > 1
        negative_prompt = negative_prompts[i:i+batch_size]
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            gerenator=generator,            
            num_images_per_prompt=batch_size,
        ).images[0]
        
        if batch_size > 1:
            for j, image in enumerate(images):
                # Calculate text size and create a new image with space for the text
                font = ImageFont.load_default()
                # Calculate text size
                text_bbox = font.getbbox(prompt[i])
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Create a new image with extra space at the top for the text
                new_image_height = image.height + text_height + 20
                new_image = Image.new('RGB', (image.width, new_image_height), (255, 255, 255))  # White background

                # Draw the original image onto the new image
                new_image.paste(image, (0, text_height + 20))

                # Draw the prompt text at the top of the new image
                draw = ImageDraw.Draw(new_image)
                draw.text((10, 10), prompt, font=font, fill=(0, 0, 0))  # Black text color

                # Save the image
                base_count = len(glob(os.path.join(OUTPUT, "*.png")))
                image_output = os.path.join(OUTPUT, f"sample_{base_count}.png")
                print(f"Will save to {image_output}")
                new_image.save(image_output)
        
    print("Done!")
    
    
if __name__ == "__main__":
    fire.Fire(main)