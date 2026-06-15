import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

DATA_DIR = r"交大视觉印象数据集2026\object_detection\data"
OUTPUT_DIR = r"text_detect_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

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
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.backbone(x)

model = SimpleDetModel().to(DEVICE)
model.load_state_dict(torch.load("text_det_model.pth", map_location=DEVICE))
model.eval()

all_files = []
for f in os.listdir(DATA_DIR):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        json_name = os.path.splitext(f)[0] + ".json"
        if os.path.exists(os.path.join(DATA_DIR, json_name)):
            all_files.append(f)

print("==== 开始推理 ====")
save_cnt = 0
for img_name in all_files:
    img_path = os.path.join(DATA_DIR, img_name)
    try:
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"跳过损坏文件：{img_name}")
        continue

    img_tensor = img_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_box = model(img_tensor)[0].cpu().numpy()

    x1 = int(np.clip(pred_box[0] * w, 0, w))
    y1 = int(np.clip(pred_box[1] * h, 0, h))
    x2 = int(np.clip(pred_box[2] * w, 0, w))
    y2 = int(np.clip(pred_box[3] * h, 0, h))

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # 不再用cv2.imwrite，转PIL保存，兼容中文路径
    res_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    save_path = os.path.join(OUTPUT_DIR, img_name)
    res_pil.save(save_path)
    save_cnt += 1

print(f"推理结束，实际成功写入{save_cnt}张图片")