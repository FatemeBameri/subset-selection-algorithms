import os
import random
import shutil
from pathlib import Path

# ================= تنظیمات =================
SRC_ROOT = Path("path/to/mvtec/bottle/train")   # مسیر دیتای اصلی
DST_ROOT = Path("path/to/mvtec_0_5_percent/bottle/train")
PERCENT = 0.005        # 0.5%
SEED = 42
# ==========================================

random.seed(SEED)

DST_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in SRC_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    n_total = len(images)

    if n_total == 0:
        continue

    n_select = max(1, int(n_total * PERCENT))
    selected = random.sample(images, n_select)

    # ساخت پوشه مقصد
    dst_class_dir = DST_ROOT / class_dir.name
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        shutil.copy(img_path, dst_class_dir / img_path.name)

    print(f"[{class_dir.name}] {n_select}/{n_total} selected")
