# ============================================================
# 1. Install Libraries
# ============================================================
!pip install ultralytics kagglehub

from ultralytics import YOLO
import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split

print("Libraries installed successfully.")


# ============================================================
# 2. Download Dataset from Kaggle
# ============================================================
print("\nDownloading PlantVillage dataset...")
dataset_path = kagglehub.dataset_download("emmarex/plantdisease")

print("Dataset downloaded to:", dataset_path)


# ============================================================
# 3. Setup Path Dataset
# ============================================================
base_dir = os.path.join(dataset_path, "PlantVillage")
output_dir = "/content/datasets/plant_yolo"

print("Base Dataset Folder:", base_dir)

# Valid image extensions
VALID_EXT = (".jpg", ".jpeg", ".png")

# Make folders
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/train", exist_ok=True)
os.makedirs(f"{output_dir}/val", exist_ok=True)


# ============================================================
# 4. Split Dataset (Train–Val)
# ============================================================
print("\nSplitting dataset into train/val...")

classes = [cls for cls in os.listdir(base_dir)
           if os.path.isdir(os.path.join(base_dir, cls))]

if not classes:
    raise ValueError("No class folders found in dataset!")

print(f"Found {len(classes)} classes.")


for cls in classes:
    class_dir = os.path.join(base_dir, cls)

    # Collect valid images only
    images = [
        os.path.join(class_dir, img)
        for img in os.listdir(class_dir)
        if img.lower().endswith(VALID_EXT)
    ]

    if len(images) == 0:
        print(f"⚠ Warning: No images found in class '{cls}'. Skipping.")
        continue

    # Split data 80% / 20%
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Make class folders
    os.makedirs(f"{output_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{output_dir}/val/{cls}", exist_ok=True)

    # Copy train images
    for img in train_imgs:
        shutil.copy(img, f"{output_dir}/train/{cls}/")

    # Copy val images
    for img in val_imgs:
        shutil.copy(img, f"{output_dir}/val/{cls}/")

    print(f"Class '{cls}' → {len(train_imgs)} train, {len(val_imgs)} val.")


print("\nDataset successfully prepared for YOLOv8 Classification!")
print("Output directory:", output_dir)


# ============================================================
# 5. Train YOLOv8 Classification Model
# ============================================================
print("\nStarting YOLOv8 classification training...")

model = YOLO("yolov8s-cls.pt")  # model: yolov8n-cls, s, m, l, x

model.train(
    data=output_dir,   # folder containing train/val subfolders
    epochs=20,         # number of training epochs
    imgsz=224,         # input image size
    batch=32,          # you can adjust if needed
)

print("\n🎉 Training completed successfully!")
