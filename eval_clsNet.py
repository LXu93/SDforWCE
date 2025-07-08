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

def evaluate_model(device, model, dataloader, classes=None):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"]
            labels = [classes.index(target) for target in labels]

            outputs = model(images)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            preds = torch.argmax(outputs, dim=1) 

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels) 

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    return all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(['Galar'], 
                            used_classes=["non-polyp","polyp"],
                            #used_classes=["bubbles","dirt","clean"],
                            phase='test',
                            #need_tensor=False,
                            #resize=False,
                            resize=(224,224),
                            augment=False,
                            #normalize_mean=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)
    #test_dataset = GeneratedDataSet("D:\Lu\data_thesis\eval_images\eval image")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classes = ["non-polyp", "polyp"]
    #classes = ["bubbles","dirt","clean"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    #model = PeftModel.from_pretrained(model, r'checkpoints\ResNet_cls_polyp2\best').to(device)
    #model = model.merge_and_unload()
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_polyp\best.pth', weights_only=True))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    evaluate_model(device, model, test_loader, classes)