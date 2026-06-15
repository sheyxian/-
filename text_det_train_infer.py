import os
import json
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===================== 路径配置 =====================
DATA_DIR = r"交大视觉印象数据集2026\object_detection\data"
OUTPUT_DIR = r"text_detect_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练超参
TRAIN_RATIO = 0.8
BATCH_SIZE = 4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {DEVICE}")

# 图像预处理
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ===================== 1. 自定义数据集（读取图片+JSON标注） =====================
class TextDetDataset(Dataset):
    def __init__(self, file_list, data_dir, transform=None):
        self.file_list = file_list
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        json_path = os.path.join(self.data_dir, os.path.splitext(img_name)[0] + ".json")

        # 读取图片
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 读取标注框，转为归一化坐标
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        boxes = []
        for shape in ann["shapes"]:
            pts = np.array(shape["points"])
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            # 归一化
            boxes.append([x1/w, y1/h, x2/w, y2/h])

        if self.transform:
            img = self.transform(img)
        # 简单取第一个框做训练（课程演示够用）
        if boxes:
            box = torch.tensor(boxes[0], dtype=torch.float32)
        else:
            box = torch.zeros(4)
        return img, box

# 加载所有样本
all_files = []
for f in os.listdir(DATA_DIR):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        json_file = os.path.splitext(f)[0] + ".json"
        if os.path.exists(os.path.join(DATA_DIR, json_file)):
            all_files.append(f)

# 划分训练/测试集
random.shuffle(all_files)
split = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:split]
val_files = all_files[split:]

train_dataset = TextDetDataset(train_files, DATA_DIR, img_transform)
val_dataset = TextDetDataset(val_files, DATA_DIR, img_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

# ===================== 2. 简易检测模型 =====================
class SimpleDetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 输出 x1,y1,x2,y2
        )
    def forward(self, x):
        return self.backbone(x)

# 初始化模型、损失、优化器
model = SimpleDetModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===================== 3. 开始训练 =====================
print("\n==== 开始训练文字检测模型 ====")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for imgs, boxes in train_loader:
        imgs = imgs.to(DEVICE)
        boxes = boxes.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, boxes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

# 保存训练好的模型权重
torch.save(model.state_dict(), "text_det_model.pth")
print("✅ 模型训练完成，权重已保存为 text_det_model.pth")

# ===================== 4. 推理：生成带红框图片（增加异常捕获） =====================
print("\n==== 开始推理，生成检测图 ====")
model.load_state_dict(torch.load("text_det_model.pth", map_location=DEVICE))
model.eval()

for img_name in all_files:
    img_path = os.path.join(DATA_DIR, img_name)

    # 优先用PIL读取，规避中文路径/损坏文件问题
    try:
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️  跳过损坏/无法读取文件: {img_name}")
        continue

    # 预处理送入模型
    img_tensor = img_transform(img_pil).unsqueeze(0).to(DEVICE)

    # 模型推理
    with torch.no_grad():
        pred_box = model(img_tensor)[0].cpu().numpy()
    # 反归一化
    x1 = int(np.clip(pred_box[0] * w, 0, w))
    y1 = int(np.clip(pred_box[1] * h, 0, h))
    x2 = int(np.clip(pred_box[2] * w, 0, w))
    y2 = int(np.clip(pred_box[3] * h, 0, h))

    # 绘制红框
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img_cv)

print("✅ 推理完成，图片已保存至 text_detect_result")