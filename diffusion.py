import os
import argparse
import random
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader, Subset
from data.dataset import *
import wandb
from tqdm import tqdm
from util import Const, standard_label, PromptHelper

def train_stable_diffusion(dataset,
                           naive_prompt=None, 
                           generated_prompt=True,
                           check_points_dir = "checkpoints", 
                           fine_tune=None,
                           epoch_num = 20,
                           batch_size = 8,
                           lr = 5e-6,
                           weight_decay=0,
                           experiment_id = "SD1.5-polyp",
                           save_safetensors = True):
    
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    num_epochs = epoch_num
    batch_size = batch_size
    learning_rate = lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {"learning_rate": learning_rate, "weight decay":weight_decay,"epochs": num_epochs, "batch_size": batch_size, "train_timestep":1000}
    wandb.init(project=experiment_id, config=config)
    

    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to("cuda")
    if fine_tune:
        pipeline.unet = UNet2DConditionModel.from_pretrained(fine_tune).to(device)
    vae = pipeline.vae 
    
    #pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, num_train_timesteps = 2000)  # Adjust total diffusion steps
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    denoise_net = pipeline.unet

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(denoise_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Training loop
    denoise_net.train()
    if not os.path.isdir(check_points_dir):
        os.makedirs(check_points_dir)
    if generated_prompt:
        prompt_generator = PromptHelper()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            pixel_values = batch["image"].to(device)
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
            #print(prompt)
            encoder_hidden_states = text_encoder(input_ids)["last_hidden_state"].to(device)

            # Encode images into latent space
            latents = vae.encode(pixel_values).latent_dist.sample() * pipeline.vae.config.scaling_factor # default scaling factor 0.18215
            
            # Generate noise
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to latents
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = denoise_net(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train loss":loss})

        if (epoch + 1) % 1 == 0:
            if save_safetensors:
                denoise_net.save_pretrained(os.path.join(check_points_dir, f"epoch_{epoch+1}"))
            else:
                torch.save(denoise_net.state_dict(), os.path.join(check_points_dir, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir')
    parser.add_argument('--interest')
    parser.add_argument('--use_dataset')
    parser.add_argument('--id', type=str, default="test")
    parser.add_argument('--uni_prompt', type=str)
    parser.add_argument('--finetune', default=None)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase', default=None)

    args = parser.parse_args()

    def sample_from_dataset(dataset, num_samples, seed=42):
      random.seed(seed)
      indices = random.sample(range(len(dataset)), num_samples)
      return Subset(dataset, indices)
    
    '''
    positive_set = prepare_data(Const.WCE_Datasets, 
                            used_classes=Const.Text_Annotation['polyp'],
                            phase=None,
                            database_dir=args.database_dir)
    negative_set = prepare_data(Const.WCE_Datasets, 
                            used_classes=Const.Text_Annotation["non-polyp"],
                            phase=None,
                            database_dir=args.database_dir)
    '''
    positive_galar = prepare_data(['Galar'], 
                            used_classes=Const.Text_Annotation['polyp'],
                            phase="train",
                            used_section=None,
                            database_dir=args.database_dir)
    negative_galar = prepare_data(['Galar'], 
                            used_classes=Const.Text_Annotation["non-polyp"],
                            phase="train",
                            used_section=None,
                            database_dir=args.database_dir)

    #positive_galar = sample_from_dataset(positive_galar, 1500, seed=42)
    #negative_galar = sample_from_dataset(negative_galar, 1500, seed=42)
    #negative_set = sample_from_dataset(negative_set, 1673, seed=42)
    dataset = ConcatDataset([positive_galar, negative_galar])
    #dataset = ConcatDataset([negative_set, positive_set, positive_galar, negative_galar])

    '''
    dataset = prepare_data(["Galar"], 
                            used_classes=None,
                            phase="train",
                            database_dir=args.database_dir)
    '''


    check_points_dir = os.path.join(os.path.abspath(os.getcwd()),"checkpoints",args.id)

 
    train_stable_diffusion(dataset, 
                           naive_prompt=False, 
                           check_points_dir= check_points_dir,
                           #fine_tune="checkpoints/bubble_dirt_section/epoch_4",
                           epoch_num=args.epoch_num,
                           batch_size=args.batch_size,
                           experiment_id=args.id)

