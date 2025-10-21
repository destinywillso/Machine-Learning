import os, json
from tqdm import tqdm
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

IMAGES_DIR = "images_trainvl"
TRAIN_JSONL = "train.jsonl"
INFO_JSON = "info.json"

if os.path.exists(INFO_JSON):
    with open(INFO_JSON, "r", encoding="utf-8") as f:
        info = json.load(f)
    print("Loaded info.json. Examples:", type(info))
else:
    print("info.json not found at", INFO_JSON)

annotations_by_image = {}
filtered_data = []

with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading train.jsonl"):
        item = json.loads(line)
        image_id = item.get("image_id")  # 或 "id"，看你的 JSON
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if os.path.exists(image_path):
            filtered_data.append(item)
        else:
            print(f"Warning: Missing image {image_id}.jpg, skipping.")

# 重建 annotations_by_image
annotations_by_image = defaultdict(lambda: {"annotations": []})
for item in filtered_data:
    image_id = item.get("image_id")
    for ann in item.get("annotations", []):
        annotations_by_image[image_id]["annotations"].append(ann)

print("Total images after filtering:", len(annotations_by_image))

def load_image_and_mask(image_id):
    image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations_by_image[image_id]["annotations"]:
        x, y, w, h = map(int, ann[0]["adjusted_bbox"]) 
        mask[y:y+h, x:x+w] = 1  

    return img, mask

image_ids = list(annotations_by_image.keys())
random.shuffle(image_ids)

train_split = image_ids[:int(0.7*len(image_ids))]
val_split = image_ids[int(0.7*len(image_ids)):int(0.85*len(image_ids))]
test_split = image_ids[int(0.85*len(image_ids)):]

img, mask = load_image_and_mask("0000189")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask (Bounding Box Regions)")
plt.show()

class CTWDataset(Dataset):
    def __init__(self, image_ids, img_dir, annotations):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.annotations = annotations

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img, mask = load_image_and_mask(image_id)
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),   # 128 -> 256
            nn.Sigmoid()  # 输出每个像素的概率
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.conv_final = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.conv_final(d1)
        return self.sigmoid(out)
    

train_dataset = CTWDataset(train_split, IMAGES_DIR, annotations_by_image)
val_dataset = CTWDataset(val_split, IMAGES_DIR, annotations_by_image)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.BCELoss()  # 二分类像素级损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5  # 先少训练试运行

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

model.eval()
with torch.no_grad():
    img, mask = val_dataset[0]
    img_tensor = img.unsqueeze(0).to(device)
    pred_mask = model(img_tensor)[0, 0].cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(img.permute(1,2,0))
plt.title("Image")

plt.subplot(1,3,2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth Mask")

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap="hot")
plt.title("Predicted Probability Mask")
plt.show()

unet_model = UNet().to(device)
optimizer_unet = torch.optim.Adam(unet_model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

num_epochs = 5

for epoch in range(num_epochs):
    unet_model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer_unet.zero_grad()
        outputs = unet_model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer_unet.step()

        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"UNet Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

unet_model.eval()
with torch.no_grad():
    img, mask = val_dataset[0]
    img_tensor = img.unsqueeze(0).to(device)
    pred_mask = unet_model(img_tensor)[0, 0].cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(img.permute(1,2,0))
plt.title("Image")

plt.subplot(1,3,2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth Mask")

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap="hot")
plt.title("UNet Predicted Mask")
plt.show()

torch.save(model.state_dict(), "simple_cnn.pth")
torch.save(unet_model.state_dict(), "unet_model.pth")