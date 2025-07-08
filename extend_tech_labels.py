from data.dataset import *
from classification import clsNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
        device = "cuda" if torch.cuda.is_available() else "cpu" 

        test_dataset = prepare_data(["Galar"], 
                                used_classes=Const.Text_Annotation['polyp'],
                                phase=None,
                                #need_tensor=False,
                                #resize=False,
                                resize=(224,224),
                                augment=False,
                                #normalize_mean=False,
                                normalize_mean=[0.485, 0.456, 0.406],
                                normalize_std=[0.229, 0.224, 0.225],
                                database_dir="D:\Lu\data_thesis")
        dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        classes = ["bubbles",
                    "dirt",
                    "clean"]

        model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)

        model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_tech_remove\best.pth', weights_only=True))

        name_results = []
        tech_results = []
        model.eval()
        model.to(device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                    names = batch["name"]
                    path = batch["image_path"]
                    labels = batch["label"]
                    sections = batch["section"]
                    image_names = [os.path.join(section,label, name) for section,label,name in zip(sections,labels,names)]
                    name_results.extend(path)
                    images = batch["image"].to(device)
                    outputs = model(images)
                    softmax = nn.Softmax(dim=1)
                    outputs = softmax(outputs)
                    preds = torch.argmax(outputs, dim=1)
                    tech_label = [classes[pred] for pred in preds]
                    tech_results.extend(tech_label)
        df = pd.DataFrame()
        df['name'] = name_results
        df['tech'] = tech_results
        df.to_csv("extended_tech_labels.csv", index=False)
        
