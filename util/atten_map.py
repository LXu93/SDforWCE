import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
import random
from tqdm import tqdm
from diffusers.models.attention_processor import AttnProcessor
from matplotlib import pyplot as plt
from daam import trace, set_seed

model_name = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_name)
pipeline.safety_checker = None
pipeline.to(device)
fine_tuned_unet = UNet2DConditionModel.from_pretrained("checkpoints/sd_polyp_simpleprompt_clen/epoch_3")
fine_tuned_unet.to(device)
pipeline.unet = fine_tuned_unet

prompt = 'Endoscopy image with polyp and bubbles.'
gen = set_seed(80)
word_list = prompt.split(' ')
#word_list = ['polyp','right']
with torch.no_grad():
    with trace(pipeline) as tc:
        out = pipeline(prompt, 
                       num_inference_steps=50, 
                       guidance_scale = 10,
                       generator=gen)
        heat_map = tc.compute_global_heat_map()
        
        num_subfigs = len(word_list) + 1
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(1, num_subfigs)
        if num_subfigs == 1:
            subfigs = [subfigs]  
     
        for i, subfig in enumerate(subfigs):
            ax = subfig.subplots()
            if i == 0:
                ax.imshow(out.images[0])
                ax.axis('off')
            else:
                word_heat_map = heat_map.compute_word_heat_map(word_list[i-1])
                word_heat_map.plot_overlay(out.images[0],ax=ax)
                ax.axis('off')
        plt.show()