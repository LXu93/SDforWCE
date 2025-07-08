import torch
from diffusers import  StableDiffusionControlNetPipeline,ControlNetModel, UNet2DConditionModel
import os
import random
import numpy as np
from safetensors.torch import load_file
from PIL import Image, ImageOps
from tqdm import tqdm
from compel import Compel
from util import PromptHelper

def inference_controlNet(prompt, 
                        control_image,
                        prompt_keys=None,
                        unet_dir="./unet_trained", 
                        controlnet_dir="./unet_trained", 
                        output_dir="output",
                        image_nums = 1,
                        num_inference_steps=50,
                        guidance_scale = 8,
                        random_gs = False,
                        gen = None):
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(controlnet_dir)
    #controlnet.load_state_dict(state_dict, strict=False)
    
    # load the original pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_name, controlnet=controlnet)
    pipeline.to(device)
    pipeline.safety_checker = None
    
    # load fine-tuned UNet model
    if unet_dir.endswith('.pth'):
        pipeline.unet.load_state_dict(torch.load(unet_dir, weights_only=True))
    else:    
        fine_tuned_unet = UNet2DConditionModel.from_pretrained(unet_dir)
        fine_tuned_unet.to(device)
        pipeline.unet = fine_tuned_unet

    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        prompt_helper = PromptHelper()
        pbar = tqdm(range(1,image_nums+1))
        for i in pbar:
            if prompt_keys:
                prompt = prompt_helper.generate_prompt_infer(**prompt_keys)
            if random_gs:
                guidance_scale = random.randrange(random_gs[0],random_gs[1])
            prompt_embeds = compel_proc(prompt)
            gen_image = pipeline(prompt_embeds=prompt_embeds, image=control_image,num_inference_steps= num_inference_steps, controlnet_conditioning_scale=2.0, guidance_scale = guidance_scale, genertor=gen).images[0]
            gen_image.save(os.path.join(output_dir, f"output_image_{i}.png"))
            pbar.set_description(f"Generated {i}/{image_nums} image(s)")
        print(f"Generated {i} images and saved in {output_dir}")

if __name__ == "__main__":
    mask = Image.open(r"C:\Users\User1\OneDrive - NTNU\Master Thesis\mask6.png")

    inference_controlNet(image_nums = 1,
                        prompt="endoscopy image from the small intestine showing polyp",
                        control_image=mask, 
                        num_inference_steps= 100,
                        guidance_scale = 10,
                        #random_gs=(6,16),
                        gen=torch.manual_seed(18),
                        unet_dir=r"checkpoints/sd_polyp_simpleprompt_clen/epoch_3",
                        controlnet_dir=r"checkpoints\polyp_controlnrt_depth\epoch_3",
                        output_dir="output/polyp_controlnet/polyp/si/")

