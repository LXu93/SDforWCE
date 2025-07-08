import argparse
from data.dataset import *
from classification import clsNet
import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import classification_report

from util import gradCAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir')
    parser.add_argument('--save_dir')
    
    args = parser.parse_args()

    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(["Galar"], 
                            used_classes=['non-polyp'],
                            phase="train",
                            #need_tensor=False,
                            resize=False,
                            #resize=(224,224),
                            augment=False,
                            #normalize_mean=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classes = ['non-polyp','polyp']
    #classes = Const.Text_Annotation["technical"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_polyp\best.pth', weights_only=True))

    save_dir = args.save_dir
    for batch in tqdm(data_loader):
        input_tensor = batch["image"].to(device)
        names = batch["name"]
        labels = batch["label"]
        sections = batch["section"]
        image_names = [os.path.join(section,label, name) for section,label,name in zip(sections,labels,names)]
        widths = batch["org_width"]
        heights = batch["org_height"]
        sizes = [(width, height) for width, height in zip(widths, heights)]
        cams, _ = gradCAM.extract_cam(device, model, input_tensor,target_class=1)

        #print(sizes[0])
        gradCAM.postprocess_and_save_cam(
            cams,
            save_dir,                  
            image_names,          
            sizes,
            threshold=0.3          
            )