import random
import shutil
from pathlib import Path

# ================= تنظیمات =================
SRC_ROOT = Path("/content/drive/MyDrive/mvtec/capsule")
DST_ROOT = Path("/content/few_mvtec/capsule")

TRAIN_PERCENT = 0.2   # 20%
SEED = 42
# ==========================================

random.seed(SEED)

# ---------- 1. کپی کامل test ----------
src_test = SRC_ROOT / "test"
dst_test = DST_ROOT / "test"

if dst_test.exists():
    shutil.rmtree(dst_test)

shutil.copytree(src_test, dst_test)
print("✅ test copied completely")

# ---------- 2. کپی کامل ground_truth ----------
src_gt = SRC_ROOT / "ground_truth"
dst_gt = DST_ROOT / "ground_truth"

if dst_gt.exists():
    shutil.rmtree(dst_gt)

shutil.copytree(src_gt, dst_gt)
print("✅ ground_truth copied completely")

# ---------- 3. سمپل‌گیری 20٪ از train ----------
src_train = SRC_ROOT / "train"
dst_train = DST_ROOT / "train"
dst_train.mkdir(parents=True, exist_ok=True)

for class_dir in src_train.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    n_total = len(images)

    if n_total == 0:
        continue

    n_select = max(1, int(n_total * TRAIN_PERCENT))
    selected = random.sample(images, n_select)

    dst_class_dir = dst_train / class_dir.name
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        shutil.copy(img_path, dst_class_dir / img_path.name)

    print(f"[train/{class_dir.name}] {n_select}/{n_total} selected")

print("🎉 Dataset creation finished!")
