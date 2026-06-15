import os
from PIL import Image
import numpy as np

# 路径配置
DETECT_IMG_DIR = r"text_detect_result"
FINAL_SAVE_DIR = r"任务二_24组可视化"

os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

# 12个地标类别
LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
GROUP_PER_CLS = 2    # 每类2组
IMG_PER_GROUP = 10    # 每组拼接10张图（2行5列）

def safe_load_img(img_path):
    try:
        return Image.open(img_path).convert("RGB")
    except:
        # 异常就返回空白白底图占位
        return Image.new("RGB", (600, 400), color=(255, 255, 255))

def uniform_pad(img, target_w, target_h):
    w, h = img.size
    new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    new_img.paste(img, ((target_w - w)//2, (target_h - h)//2))
    return new_img

def stitch_2row5(img_list):
    # 补齐到10张，多余截断
    while len(img_list) < IMG_PER_GROUP:
        img_list.append(Image.new("RGB", (600, 400), (255, 255, 255)))
    img_list = img_list[:IMG_PER_GROUP]

    max_w = max(i.width for i in img_list)
    max_h = max(i.height for i in img_list)
    std_imgs = [uniform_pad(i, max_w, max_h) for i in img_list]

    row1 = np.hstack([np.array(i) for i in std_imgs[:5]])
    row2 = np.hstack([np.array(i) for i in std_imgs[5:]])
    full_img = np.vstack([row1, row2])
    return Image.fromarray(full_img)

# 批量遍历每一个地标
for cls_name in LANDMARKS:
    img_names = []
    for fname in os.listdir(DETECT_IMG_DIR):
        if fname.startswith(cls_name) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_names.append(fname)
    img_names.sort()
    total = len(img_names)
    print(f"\n[{cls_name}] 检索到图片数量：{total}")

    group_idx = 0
    for start_idx in range(0, total, IMG_PER_GROUP):
        if group_idx >= GROUP_PER_CLS:
            break
        group_idx += 1
        end_idx = start_idx + IMG_PER_GROUP
        batch_names = img_names[start_idx:end_idx]

        imgs = [safe_load_img(os.path.join(DETECT_IMG_DIR, n)) for n in batch_names]
        combined = stitch_2row5(imgs)

        save_name = f"{cls_name}_第{group_idx}组_检测可视化.jpg"
        save_full_path = os.path.join(FINAL_SAVE_DIR, save_name)
        combined.save(save_full_path, quality=95)
        print(f"已生成：{save_name}")

print("\n======================================")
print("✅ 全部24组拼接图生成完毕！")
print(f"输出目录：{FINAL_SAVE_DIR}")