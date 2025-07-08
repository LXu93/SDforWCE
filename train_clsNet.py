import argparse
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data.dataset import *
from classification import clsNet
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir')
    parser.add_argument('--id', type=str, default="ResNet_cls_aug_baseline")
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    id = args.id
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    transform = transforms.Compose([
                              transforms.Resize((224,224),interpolation=InterpolationMode.BILINEAR),
                              transforms.RandomHorizontalFlip(p=0.3),
                              transforms.RandomVerticalFlip(p=0.3),
                              transforms.RandomRotation(degrees=(-30,30)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])
    train_dataset = prepare_data(["Galar"], 
                            used_classes=None,
                            phase="train",
                            resize=(224,224),
                            augment=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)
    
    test_dataset = prepare_data(["Galar"], 
                            used_classes=None,
                            phase="test",
                            resize=(224,224),
                            augment=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir=args.database_dir)

    check_points_dir = os.path.join(os.path.abspath(os.getcwd()),"checkpoints", id)
    clsNet.train(device, train_dataset, test_dataset, 
                 #classes=Const.Text_Annotation["technical"],
                 classes=["non-polyp", "polyp"],
                 #classes=["bubbles","dirt","clean"],
                 model_name='ResNet',
                 freeze_num= 7,
                 lr = 1e-4,
                 weight_decay= 0,
                 ckp_dir=check_points_dir,
                 batch_size_train=args.batch_size,
                 num_epochs=args.epoch_num)