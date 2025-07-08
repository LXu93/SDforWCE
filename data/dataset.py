import torch
import pandas as pd
import random
import os
from PIL import Image
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from util import Const, standard_label

def sample_from_dataset(dataset, num_samples, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, indices)

def read_path_and_labels(csv_file_path, used_classes = None, phase = None, used_section=None):
    df = pd.read_csv(csv_file_path)
    if phase is not None:
      df = df[df["Split"]==phase]
    if used_classes is not None:
      df = df[df["Label"].isin(used_classes)]
    section_list = []
    if "Section" in df.columns:
      if used_section is not None:
        df = df[df["Section"].isin(used_section)]
      section_list = df["Section"].tolist()
    images_path_list = df["Image Path"].tolist()
    label_list = df["Label"].tolist()
    feature_list = []
    position_list = []
    tech_list = []
    if "count" in df.columns:
      feature_list = df["count"].tolist()
      def covert_feature(count):
          if count == "unknown":
             return "unknown"
          if int(float(count)) == 0:
             return "possible"
          if int(float(count)) == 1:
             return "single"
          if int(float(count)) > 1:
             return "multiple"
      feature_list = [covert_feature(feature) for feature in feature_list]
    if "position" in df.columns:
       position_list = df["position"].tolist()
    if "tech" in df.columns:
       tech_list = df["tech"].tolist()
    return images_path_list, label_list, section_list, feature_list, position_list, tech_list

class MyDataSet(torch.utils.data.Dataset):
  def __init__(self, image_path_list, label_list, section_list = [], feature_list = [], position_list = [], tech_list=[], org_size = None, transform=None):
    super().__init__()
    self.image_path_list = image_path_list
    self.label_list = label_list
    self.section_list = section_list
    self.feature_list = feature_list
    self.position_list = position_list
    self.tech_list = tech_list
    self.transform = transform
    self.org_size = org_size

  def __len__(self):
    return len(self.image_path_list)

  def __getitem__(self, idx):
    image = Image.open(self.image_path_list[idx])
    if not self.org_size:
      width, height = image.size
    else:
      width, height = self.org_size
    image_path = self.image_path_list[idx]
    image_name = os.path.basename(self.image_path_list[idx])
    label = self.label_list[idx]
    if self.section_list:
      section = self.section_list[idx]
    else:
      section = "unknown"
    if self.feature_list:
      feature = self.feature_list[idx]
    else:
      feature = "unknown"
    if self.position_list:
      position = self.position_list[idx]
    else:
      position = "unknown"
    if self.tech_list:
      tech_label = self.tech_list[idx]
    else:
      tech_label = "unknown"
    if self.transform:
      image = self.transform(image)
    return {"image": image, 
            "label": label, 
            "section": section,
            "feature":feature,
            "position":position,
            "tech_label":tech_label, 
            "name": image_name, 
            "image_path": image_path,
            "org_width": width, 
            "org_height": height}
  
def prepare_data(dataset_list, 
                 used_classes = None, 
                 need_tensor = True,
                 augment = False,
                 normalize_mean = [0.5]*3, 
                 normalize_std = [0.5] *3,
                 resize=(320,320), 
                 database_dir="D:\Lu\data_thesis",
                 phase=None,
                 used_section = None):
    database_dir = database_dir
    data_summary = pd.read_excel(os.path.join(database_dir,"dataset_summary.xlsx"))
    datasets = []
    for dataset in dataset_list:
        dataset_dir = data_summary[data_summary["Name"]==dataset]["Dataset dir"].item()
        dataset_dir = os.path.join(*dataset_dir.split("\\")) # adapt Linux
        csv_path = data_summary[data_summary["Name"]==dataset]["Label File"].item()
        csv_path = os.path.join(*csv_path.split("\\")) # adapt Linux
        image_path_list, label_list, section_list, feature_list, position_list, tech_list = read_path_and_labels(os.path.join(database_dir,dataset_dir,csv_path), used_classes, phase, used_section)
        image_path_list = [os.path.join(database_dir,dataset_dir,*image_path.split("\\")) for image_path in image_path_list]
        preprocess_list = []
        org_size = None
        if dataset in Const.CROP.keys():
           preprocess_list.append(transforms.CenterCrop(Const.CROP[dataset]))
           org_size = Const.CROP[dataset]
        if resize:
           preprocess_list.append(transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR))
        if augment:
           preprocess_list += [transforms.RandomHorizontalFlip(p=0.3),
                               transforms.RandomVerticalFlip(p=0.3),
                              transforms.RandomRotation(degrees=(-30,30))]
        if need_tensor:
           preprocess_list.append(transforms.ToTensor())
        if normalize_mean and normalize_std:
           preprocess_list.append(transforms.Normalize(normalize_mean, normalize_std))
        preprocess = transforms.Compose(preprocess_list)
        datasets.append(MyDataSet(image_path_list, label_list, section_list, feature_list, position_list, tech_list, org_size=org_size, transform=preprocess))
    dataset = ConcatDataset(datasets)
    return dataset

class DataSetwithMask(torch.utils.data.Dataset):
  @staticmethod
  def generate_empty_mask(w, h):
        return Image.new('L', (w,h))
  
  def __init__(self, 
               image_dir, 
               mask_dir, 
               label_csv, # table for masks
               phase=None, 
               need_mask=['polyp'], 
               resize=None, 
               image_transform=None, 
               mask_transform=None):
    super().__init__()
    if not label_csv:
       label_csv = os.path.join(image_dir,'labels_table.csv')
    self.file_path_list, self.label_list, self.section_list, self.feature_list, self.position_list, self.tech_list = read_path_and_labels(label_csv, used_classes=None, phase=phase, used_section=None)
    self.need_mask = need_mask
    self.resize = resize
    self.image_transform = image_transform
    self.mask_transform = mask_transform


    self.image_path_list = [os.path.join(image_dir,*file_path.split("\\")) for file_path in self.file_path_list]
    self.mask_path_list = [os.path.join(mask_dir,*file_path.split("\\")) for file_path in self.file_path_list]

  def __len__(self):
    return len(self.image_path_list)

  def __getitem__(self, idx):
    image = Image.open(self.image_path_list[idx])
    width, height = image.size
    image_name = os.path.basename(self.image_path_list[idx])
    image_path = self.image_path_list[idx]

    label = self.label_list[idx]
    
    if label in self.need_mask:
      mask = Image.open(self.mask_path_list[idx])
    else:
      mask = DataSetwithMask.generate_empty_mask(width, height)

    if self.section_list:
      section = self.section_list[idx]
    else:
      section = "unknown"
    if self.feature_list:
      feature = self.feature_list[idx]
    else:
      feature = "unknown"
    if self.position_list:
      position = self.position_list[idx]
    else:
      position = "unknown"
    if self.tech_list:
      tech_label = self.tech_list[idx]
    else:
      tech_label = "unknown"

    if self.resize:
       resize_transform = transforms.Resize(self.resize, interpolation=InterpolationMode.BILINEAR)
       image = resize_transform(image)
       mask = resize_transform(mask)
    if self.image_transform:
      image = self.image_transform(image)
    if self.mask_transform:
      mask = self.mask_transform(mask)
    return {"image": image, 
            "label": label, 
            "mask": mask,
            "section": section,
            "feature":feature,
            "position":position,
            "tech_label":tech_label, 
            "name": image_name, 
            "image_path": image_path, 
            "org_width": width, 
            "org_height": height}
  
class GeneratedDataSet(torch.utils.data.Dataset):
  def __init__(self, image_dir, label="", transform=None):
    super().__init__()
    self.image_dir = image_dir
    self.label = label
    self.transform = transform
    self._image_path_list = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                self._image_path_list.append(os.path.join(root, file))

  def __len__(self):
    return len(self._image_path_list)

  def __getitem__(self, idx):
    image_path = self._image_path_list[idx]
    image = Image.open(self._image_path_list[idx])
    width, height = image.size
    if self.transform:
      image = self.transform(image)
    return {"image": image, 
            "label":self.label, 
            "section": "",
            "feature":"",
            "position":"",
            "tech_label":"", 
            "name": "",
            "image_path":image_path, 
            "org_width": width, 
            "org_height": height}

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  
  dataset = prepare_data(['Galar'], 
                            used_classes=Const.Text_Annotation['polyp'],
                            used_section=None,
                            phase='test')#Const.Text_Annotation["technical"])
  print(len(dataset))
