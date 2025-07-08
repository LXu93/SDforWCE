import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
import random
from tqdm import tqdm
from compel import Compel
import wandb
from util import prompt_generator, PromptHelper

def inference_diffusion(prompt="",
                        prompt_keys = {}, 
                        #pathology = None,
                        trained_model_dir="./unet_trained", 
                        output_dir="output",
                        image_nums = 1,
                        num_inference_steps=50,
                        guidance_scale = 8,
                        random_gs = False,
                        gen = None):
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the original pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to(device)
    pipeline.safety_checker = None

    # load fine-tuned UNet model
    if trained_model_dir.endswith('.pth'):
        pipeline.unet.load_state_dict(torch.load(trained_model_dir, weights_only=True))
    else:    
        fine_tuned_unet = UNet2DConditionModel.from_pretrained(trained_model_dir)
        fine_tuned_unet.to(device)
        pipeline.unet = fine_tuned_unet

    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

    '''
    wandb.init(
        project="generation_evaluation",  
        name="ext epoch4", 
        config={
            "weight_decay":0,
            "checkpoint":trained_model_dir,
        }
    )
    '''
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    prompt_helper = PromptHelper()
    with torch.no_grad():
        pbar = tqdm(range(1,image_nums+1))
        images = []
        prompts = []
        scales = []
        for i in pbar:
            if prompt_keys:
                prompt = prompt_helper.generate_prompt_infer(**prompt_keys)
            #if pathology:
            #    prompt = prompt_generator(pathology, section="unknown")
            if random_gs:
                guidance_scale = random.randrange(random_gs[0],random_gs[1])
            #print(prompt)
            prompt_embeds = compel_proc(prompt)
            image = pipeline(prompt_embeds=prompt_embeds, num_inference_steps= num_inference_steps, guidance_scale = guidance_scale, genertor=gen).images[0]
            images.append(image)
            prompts.append(prompt)
            scales.append(guidance_scale)
            #image.save(os.path.join(output_dir, f"output_image_{i}.png"))
            image.save(os.path.join(output_dir, f"{i}_{prompt}.png"))
            pbar.set_description(f"Generated {i}/{image_nums} image(s)")
        #table = wandb.Table(columns=["Prompt", "Scale", "Image"])
        #for prompt, scale, img in zip(prompts, scales, images):
        #    table.add_data(prompt, scale, wandb.Image(img, caption=f"{prompt} | {scale}"))
        #wandb.log({"generation_gallery": table})
        #wandb.finish()
        print(f"Generated {i} images and saved in {output_dir}")

if __name__ == "__main__":
    inference_diffusion(image_nums = 1,
                        prompt='An endoscopy image from the small intestine shows a polyp.',
                        prompt_keys={}, 
                        num_inference_steps= 100,
                        guidance_scale = 8,
                        #random_gs=(4,12),
                        gen=torch.manual_seed(88),
                        trained_model_dir="checkpoints/sd_polyp_simpleprompt_clen/epoch_3",
                        output_dir="output/time_step")

