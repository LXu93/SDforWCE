import argparse
from data.dataset import *
from classification import clsNet
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


from util import gradCAM


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(["Galar"], 
                            used_classes=Const.Text_Annotation['polyp'],
                            phase=None,
                            #need_tensor=False,
                            resize=False,
                            #resize=(224,224),
                            augment=False,
                            #normalize_mean=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir="D:\Lu\data_thesis")
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classes = ['non-polyp','polyp']
    #classes = Const.Text_Annotation["technical"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_combined\best.pth', weights_only=True))

    count_results = []
    names_results = []
    position_results = []
    for batch in tqdm(data_loader):
        input_tensor = batch["image"].to(device)
        names = batch["name"]
        path = batch["image_path"]
        labels = batch["label"]
        sections = batch["section"]
        image_names = [os.path.join(section,label, name) for section,label,name in zip(sections,labels,names)]
        widths = batch["org_width"]
        heights = batch["org_height"]
        sizes = [(int(width), int(height)) for width, height in zip(widths, heights)]
        cams, _ = gradCAM.extract_cam(device, model, input_tensor,target_class=1)

        B = cams.shape[0]

        for i in range(B):
            cam = cams[i].unsqueeze(0)
            cam_np = gradCAM.postprocess_attribution(cam, sizes[i], threshold=0)
            try:
                count, cnt = gradCAM.count_by_contour(cam_np, threshold=0.6, min_area=60)
                count_results.append(count)
            except Exception as e:
                count = 0
                count_results.append(count)
            if count == 1:
                position = gradCAM.get_position_3x3(cnt[0],sizes[i])
                position_results.append(position)
            else:
                position_results.append("unknown")
            names_results.append(path[i])
    df = pd.DataFrame()
    df['name'] = names_results
    df['count'] = count_results
    df['position'] = position_results
    df.to_csv("extended_labels.csv", index=False)
    