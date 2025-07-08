import torch
_ = torch.manual_seed(123)
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import random
from data.dataset import *


def get_kid_score(real_dataloader, fake_dataloader):
    kid = KernelInceptionDistance(subset_size=50)
    for real_samples in tqdm(real_dataloader):
        real_images = real_samples["image"].mul(255).to(torch.uint8)
        kid.update(real_images, real=True)
    for fake_samples in tqdm(fake_dataloader):
        fake_images = fake_samples["image"].mul(255).to(torch.uint8)
        kid.update(fake_images, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std

def get_fid_score(real_dataloader, fake_dataloader):
    fid = FrechetInceptionDistance(feature=2048)
    for real_samples in tqdm(real_dataloader):
        real_images = real_samples["image"].mul(255).to(torch.uint8)
        fid.update(real_images, real=True)
    for fake_samples in tqdm(fake_dataloader):
        fake_images = fake_samples["image"].mul(255).to(torch.uint8)
        fid.update(fake_images, real=False)
    fid_score = fid.compute()
    
    return fid_score


if __name__ == "__main__":
    def sample_from_dataset(dataset, num_samples, seed=42):
      random.seed(seed)
      indices = random.sample(range(len(dataset)), num_samples)
      return Subset(dataset, indices)
    dataset_train = prepare_data(['Galar'], 
                           used_classes=['polyp'],
                           phase='train', 
                           normalize_mean=False, 
                           resize=(299,299))
    dataset_negative = prepare_data(['Galar'], 
                           used_classes=['non-polyp'],
                           phase='train', 
                           normalize_mean=False, 
                           resize=(299,299))
    dataset_remove = prepare_data(['Galar'], 
                        used_classes=['polyp'],
                        phase='not_used', 
                        normalize_mean=False, 
                        resize=(299,299))
    dataset = ConcatDataset([dataset_train,dataset_remove])

    positive_set = prepare_data(Const.WCE_Datasets, 
                            used_classes=Const.Text_Annotation['polyp'],
                            phase=None,
                            normalize_mean=False, 
                            resize=(299,299))
    negative_set = prepare_data(Const.WCE_Datasets, 
                            used_classes=Const.Text_Annotation["non-polyp"],
                            phase=None,                        
                            normalize_mean=False, 
                            resize=(299,299))


    positive_galar = sample_from_dataset(dataset_train, 1500, seed=42)
    negative_galar = sample_from_dataset(dataset_negative, 1500, seed=42)
    negative_set = sample_from_dataset(negative_set, 1673, seed=42)
    dataset_combined_positive = ConcatDataset([positive_set, positive_galar])
    dataset_combined_negative = ConcatDataset([negative_set, negative_galar])

    dataset = sample_from_dataset(dataset_train, 200, seed=42)

    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        ])
    generated_data = GeneratedDataSet(r"output/experiment/epoch_7",transform=transform)
    #generated_data = sample_from_dataset(generated_data, 2000, seed=42)


    real_images_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    fake_images_loader = DataLoader(generated_data, batch_size=256, shuffle=False, drop_last=False)

    print(get_kid_score(real_images_loader, fake_images_loader))
    #print(get_fid_score(real_images_loader, fake_images_loader))