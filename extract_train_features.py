import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import wide_resnet50_2
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "mvtec_original"
features_save_root = "features"
os.makedirs(features_save_root, exist_ok=True)


backbone = wide_resnet50_2(weights='IMAGENET1K_V1').to(device)
backbone.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Projection layers برای رسیدن به 512 کانال
proj_layer2 = nn.Conv2d(256, 512, kernel_size=1).to(device)
proj_layer3 = nn.Conv2d(512, 512, kernel_size=1).to(device)

def extract_feature(img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        x = backbone.conv1(img_tensor)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        layer2 = backbone.layer1(x)  # 256 کانال
        layer3 = backbone.layer2(layer2)  # 512 کانال

        # Projection
        layer2_proj = proj_layer2(layer2)
        layer3_proj = proj_layer3(layer3)

        # Upsample layer2 به اندازه spatial layer3
        layer2_up = F.interpolate(layer2_proj, size=layer3_proj.shape[2:], mode='bilinear', align_corners=False)

        # Concat
        concat_features = torch.cat([layer2_up, layer3_proj], dim=1)  # 512+512=1024 کانال
        feature_vector = concat_features.mean(dim=[2,3])  # global avg pooling

    return feature_vector.squeeze(0).cpu().numpy()  # [1024,]


for class_name in os.listdir(dataset_root):
    class_train_dir = os.path.join(dataset_root, class_name, "train")
    if not os.path.isdir(class_train_dir):
        continue

    class_features_dir = os.path.join(features_save_root, class_name)
    os.makedirs(class_features_dir, exist_ok=True)

    train_dir = os.path.join(class_train_dir, "good")
    if not os.path.isdir(train_dir):
        continue

    image_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Found {len(image_files)} images in class {class_name}")

    for img_name in tqdm(image_files, desc=f"Processing {class_name}"):
        img_path = os.path.join(train_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)

        feature_vector = extract_feature(img_tensor)

        feature_file = os.path.join(class_features_dir, os.path.splitext(img_name)[0] + ".npy")
        np.save(feature_file, feature_vector)
