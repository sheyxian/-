import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 路径 =====================
BASE_DIR = r"交大视觉印象数据集2026\image_retrieval\base"
QUERY_DIR = r"交大视觉印象数据集2026\image_retrieval\query"
CURVE_SAVE_DIR = r"任务一_PK曲线"
os.makedirs(CURVE_SAVE_DIR, exist_ok=True)

# 配置
LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
K_LIST = [20, 40, 60]
VALID_SUFFIX = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 模型初始化
device = torch.device("cpu")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
feature_extractor.to(device)


def extract_feature(img_path):
    """单图提取特征，损坏文件直接返回空特征"""
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(img_tensor)
        return feat.squeeze().cpu().numpy()
    except Exception:
        return None


def get_all_images(root_dir):
    """递归遍历，只保留能正常打开的图片"""
    img_paths = []
    img_names = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(VALID_SUFFIX):
                full_path = os.path.join(root, f)
                # 先尝试打开校验文件合法性
                try:
                    Image.open(full_path)
                    img_paths.append(full_path)
                    img_names.append(f)
                except Exception:
                    continue
    return img_paths, img_names


# 1. 读取base所有合法图片
print("==== 正在递归读取 base 文件夹（含 BJTU / util_pic） ====")
base_img_paths, base_file_list = get_all_images(BASE_DIR)
base_feature_list = []

for path in base_img_paths:
    feat = extract_feature(path)
    if feat is not None:
        base_feature_list.append(feat)

base_num = len(base_file_list)
print(f"base 有效图片总数：{base_num}")

if base_num == 0:
    print("❌ 未找到任何有效图片，程序退出")
    exit()

base_feature_arr = np.array(base_feature_list)

# 2. 逐类检索、计算P@K、绘图
for land in LANDMARKS:
    print(f"\n---------- 处理地标：{land} ----------")
    query_file_list = []
    for f in os.listdir(QUERY_DIR):
        if f.endswith(VALID_SUFFIX) and f.startswith(land):
            query_file_list.append(f)

    if len(query_file_list) == 0:
        print(f"{land} 无查询图片，跳过")
        continue

    precision_list = []
    for k in K_LIST:
        total_hit = 0
        total_query = len(query_file_list)

        for q_name in query_file_list:
            q_path = os.path.join(QUERY_DIR, q_name)
            q_feat = extract_feature(q_path)
            if q_feat is None:
                continue
            q_feat = q_feat.reshape(1, -1)
            sim = cosine_similarity(q_feat, base_feature_arr)[0]
            topk_idx = np.argsort(-sim)[:k]

            hit = 0
            for idx in topk_idx:
                if base_file_list[idx].startswith(land):
                    hit += 1
            total_hit += hit

        avg_prec = total_hit / (total_query * k)
        precision_list.append(avg_prec)

    # 绘图保存
    plt.figure(figsize=(7, 4))
    plt.plot(K_LIST, precision_list, marker="o", linewidth=2)
    plt.title(f"{land} P@K Curve")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(CURVE_SAVE_DIR, f"{land}_P@K.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {land} 曲线保存完成")

print("\n========================================")
print("🎉 任务一全部执行完毕！12张P@K曲线已生成")
print(f"输出目录：{CURVE_SAVE_DIR}")