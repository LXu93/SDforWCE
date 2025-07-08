import os
import argparse
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from torch.utils.data import DataLoader
from data.dataset import *
import wandb
from tqdm import tqdm
from util import Const, PromptHelper, standard_label

def train_controlnet(dataset,
                           naive_prompt=False, 
                           generated_prompt=True,
                           generated_prompt_template = "",
                           check_points_dir = "checkpoints", 
                           fine_tune=None,
                           epoch_num = 20,
                           batch_size = 8,
                           lr = 1e-5,
                           experiment_id = "controlNet",
                           save_safetensors = True):
    
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    num_epochs = epoch_num
    batch_size = batch_size
    learning_rate = lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size, "train_timestep":1000}
    wandb.init(project=experiment_id, config=config)

    if fine_tune:
        pretrained_unet = UNet2DConditionModel.from_pretrained(fine_tune).to(device)
    else:
        pretrained_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # Initialize controlnet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_name, controlnet = controlnet)
    if fine_tune:
        pipeline.unet = pretrained_unet
    pipeline.to("cuda")
    
    vae = pipeline.vae 
    #pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, num_train_timesteps = 2000)  # Adjust total diffusion steps
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    denoise_net = pipeline.unet

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=learning_rate)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    denoise_net.requires_grad_(False)
    denoise_net.eval()
    
    controlnet.train()
    if not os.path.isdir(check_points_dir):
        os.makedirs(check_points_dir)
    if generated_prompt:
        prompt_generator = PromptHelper()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            pixel_values = batch["image"].to(device)
            controlnet_image = batch["mask"].to(device)
            controlnet_image = controlnet_image.repeat(1, 3, 1, 1)
            if naive_prompt:
                standardized_labels = [standard_label(label) for label in batch["label"]]
                prompt = [f"An endoscopy image shows {standard_label}" for standard_label in standardized_labels]
                #print(prompt[0])
                input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
            elif generated_prompt:
                standardized_labels = [standard_label(label) for label in batch["label"]]
                section_names = [' '.join(section.split('_')) for section in batch["section"]]
                prompt = [prompt_generator.generate_prompt(section, label, feature, position, tech) for label, section, feature, position, tech in zip(standardized_labels, section_names, batch["feature"], batch["position"], batch["tech_label"])]
                input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
            else:
                input_ids = tokenizer(batch["label"], padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)

            # Encode text
            encoder_hidden_states = text_encoder(input_ids)["last_hidden_state"].to(device)

            # Encode images into latent space
            latents = vae.encode(pixel_values).latent_dist.sample() * pipeline.vae.config.scaling_factor # default scaling factor 0.18215
            
            # Generate noise
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to latents
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

            noise_pred = denoise_net(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train loss":loss})

        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            if save_safetensors:
                #denoise_net.save_pretrained(os.path.join(check_points_dir, 'unet',f"epoch_{epoch+1}"))
                controlnet.save_pretrained(os.path.join(check_points_dir,f"epoch_{epoch+1}"))

            else:
                torch.save(controlnet.state_dict(), os.path.join(check_points_dir, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir')
    parser.add_argument('--interest', default="technical")
    parser.add_argument('--use_dataset')
    parser.add_argument('--id', type=str, default="test")
    parser.add_argument('--uni_prompt', type=str)
    parser.add_argument('--finetune', default=None)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', default=None)

    args = parser.parse_args()

    imaeg_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    dataset = DataSetwithMask(image_dir="D:\Lu\data_thesis\Galar_custo", #"/cluster/home/luxu/data_thesis/Galar_custo", 
               mask_dir= "D:\Lu\data_thesis\galar_CAM", #"/cluster/home/luxu/data_thesis/galar_CAM", 
               label_csv=None, 
               phase='train', 
               need_mask=['polyp'], 
               resize=(320,320), 
               image_transform=imaeg_transform, 
               mask_transform=mask_transform)
    
    check_points_dir = os.path.join(os.path.abspath(os.getcwd()),"checkpoints",args.id)

    train_controlnet(dataset, 
                           generated_prompt=True,
                           check_points_dir= check_points_dir,
                           epoch_num=args.epoch_num,
                           batch_size=args.batch_size,
                           experiment_id=args.id)
