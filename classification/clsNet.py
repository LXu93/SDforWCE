import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from torch.utils.tensorboard import SummaryWriter
#import wandb
import sys
sys.path.append(dir(sys.path[0]))
from util import Const, standard_label, prompt_generator

class MLP(nn.Module):
    def __init__(self, input_size, num_classes ,drop_out=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, input_size//4)
        self.fc3 = nn.Linear(input_size//4, num_classes)
        self.active = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ResNetClsNet(nn.Module): 
    def __init__(self, num_classes, fine_tune=True, freeze_num=8, drop_out=0.2):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights, progress=True)
        self.preprocess = weights.transforms()

        self.resnet.fc = nn.Identity()

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif freeze_num > 0:
            # Freeze early layers (conv1 and some blocks in layer1/layer2)
            for param in self.resnet.conv1.parameters():
                param.requires_grad = False
            
            for param in self.resnet.bn1.parameters():
                param.requires_grad = False

            # Freeze the first `freeze_num` blocks of ResNet
            layer_list = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
            num_frozen = 0
            for layer in layer_list:
                for block in layer:
                    if num_frozen < freeze_num:
                        for param in block.parameters():
                            param.requires_grad = False
                        num_frozen += 1
                    else:
                        break
                if num_frozen >= freeze_num:
                    break

        self.head = MLP(2048, num_classes, drop_out)  # ResNet-50's last layer output is 2048 features
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.head(x)
        return x

class ViTClsNet(nn.Module):
    def __init__(self, num_classes, fine_tune = True, freeze_num = 8, drop_out = 0.2):
        super().__init__()
        weights = models.ViT_L_16_Weights.DEFAULT
        self.vit = models.vit_l_16(weights=weights, progress=True)
        self.preprocess = weights.transforms()

        self.vit.heads = torch.nn.Identity()

        if drop_out and drop_out!=0:
            for layer in self.vit.encoder.layers:
                layer.dropout.p = drop_out

        if not fine_tune:
            for param in self.vit.parameters():
                param.requires_grad = False
        elif freeze_num > 0:
            for param in self.vit.conv_proj.parameters():  
                param.requires_grad = False

            for i in range(freeze_num):
                for param in self.vit.encoder.layers[i].parameters():
                    param.requires_grad = False

        self.head = MLP(1024, num_classes, drop_out)

    def forward(self, x):
        x = self.vit(x)
        x = self.head(x)
        return x
    
def train(device, 
          train_set, 
          val_set,
          #run_id = "classification",
          classes = ["non-polyp", "polyp"],
          model_name="ViT",
          lora = False, 
          lr = 5e-6,
          batch_size_train = 64,
          batch_size_test = 128,
          num_epochs = 200,
          weight_decay = 1e-4,
          freeze_num = 8,
          drop_out = 0.2,
          ckp_dir = "checkpoints/"):
    num_classes = len(classes)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_test, shuffle=False, drop_last=False)
    
    if model_name == 'ViT':
        model = ViTClsNet(num_classes, fine_tune=True, freeze_num=freeze_num, drop_out=drop_out).to(device)
        if lora:
            config = LoraConfig(
                r=8,  # Low-rank dimension (small r = less overfitting)
                lora_alpha=32,  # Scaling factor
                lora_dropout=0.1,  # Dropout for LoRA adaptation
                target_modules=["self_attention"],  # Apply LoRA only to attention layers
            )

            # Apply LoRA to Model
            model = get_peft_model(model, config).to(device)
            #model.print_trainable_parameters() 
    if model_name == 'ResNet':
        model = ResNetClsNet(num_classes, fine_tune=True, freeze_num=freeze_num, drop_out=drop_out).to(device)
        if lora:
            config = LoraConfig(
                inference_mode=False,
                r=32,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["conv1", "conv2","conv3"]
            )
            model = get_peft_model(model, config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    writer = SummaryWriter()
    writer.add_scalar('hyperparameter/learning rate',lr)
    writer.add_scalar('hyperparameter/batch size', batch_size_train)
    writer.add_scalar('hyperparameter/weight decay', weight_decay)
    writer.add_scalar('hyperparameter/freeze block num', freeze_num)
    writer.add_scalar('hyperparameter/weight decay', weight_decay)
    writer.add_scalar('hyperparameter/drop out', drop_out)
    writer.flush()
    #run = wandb.init(project=run_id)
    #wandb.config = {"learning_rate": lr, "epochs": num_epochs, "batch_size_train": batch_size_train}
    if not os.path.isdir(ckp_dir):
        os.makedirs(ckp_dir)
    
    best_loss = float('inf')
    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        mean_train_loss = 0.0
        pbar = tqdm(train_loader)
        for batch in pbar:
            
            labels = [standard_label(label) for label in batch["label"]]
            labels = [classes.index(target) for target in labels]
            labels = torch.tensor(labels,dtype=torch.long, device=device)
            images = batch["image"].to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += loss.item()
            iteration += 1
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_train_loss/len(train_loader):.4f}")
        mean_train_loss = mean_train_loss/len(train_loader)
        writer.add_scalar("Loss/Train Loss", mean_train_loss, epoch)
        writer.flush()
        #wandb.log({"train loss":mean_train_loss})
        if (epoch+1)%1 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_labels = [standard_label(label) for label in val_batch["label"]]
                    val_labels = [classes.index(target) for target in val_labels]
                    val_labels = torch.tensor(val_labels,dtype=torch.long, device=device)
                    val_images= val_batch["image"].to(device)
            
                    pre = model(val_images)
                    
                    batch_loss = criterion(pre, val_labels)
                    val_loss = val_loss + batch_loss.item()
            val_loss = val_loss/len(val_loader)
            writer.add_scalar("Loss/Val Loss", val_loss, epoch)
            writer.flush()
            #wandb.log({"Test Loss": val_loss})
            if val_loss<best_loss:
                best_loss = val_loss
                if lora:
                    model.save_pretrained(os.path.join(ckp_dir, "best"))
                else:
                    torch.save(model.state_dict(), os.path.join(ckp_dir, "best.pth"))
        if (epoch+1)%1 == 0:
            if lora:
                model.save_pretrained(os.path.join(ckp_dir, f"epoch_{epoch+1}"))
            else:
                torch.save(model.state_dict(), os.path.join(ckp_dir,f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    model = ResNetClsNet(2, fine_tune=True, freeze_num=0) 
    print(model)


    