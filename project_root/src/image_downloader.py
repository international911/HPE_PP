from bing_image_downloader import downloader
from PIL import Image, ImageEnhance, ImageOps
import imagehash
import numpy as np
import os
import shutil
import random

# Константы
MIN_SIZE = (300, 300)
DOWNLOAD_LIMIT = 100
TARGET_COUNT = 300
CLASSES = {
    "standing": "person standing full body",
    "sitting": "person sitting full body",
    "lying": "person lying down full body"
}

# Используем абсолютные пути
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Путь к project_root
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
PREVIEW_HTML = os.path.join(PROJECT_ROOT, "data", "preview.html")

def download_images():
    for class_name, query in CLASSES.items():
        output_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[🔽] Downloading images for '{class_name}'...")
        downloader.download(
            query,
            limit=DOWNLOAD_LIMIT,
            output_dir=DATASET_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )

        # Перемещаем из dataset/<class>/images/ -> dataset/<class>/
        src_dir = os.path.join(DATASET_DIR, class_name, "images")
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(DATASET_DIR, class_name, f)
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"[⚠️] Failed to move {src_path}: {e}")
            try:
                os.rmdir(src_dir)
            except Exception as e:
                print(f"[⚠️] Failed to remove directory {src_dir}: {e}")

def clean_images():
    print("[🧹] Cleaning images...")
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"[⚠️] Directory {class_dir} does not exist, skipping.")
            continue

        hashes = set()
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
                        os.remove(img_path)
                        continue

                    img_hash = imagehash.average_hash(img)
                    if img_hash in hashes:
                        os.remove(img_path)
                        continue
                    hashes.add(img_hash)

                    # Конвертация в .jpg
                    if not img_name.lower().endswith(".jpg"):
                        rgb_img = img.convert("RGB")
                        new_name = os.path.splitext(img_name)[0] + ".jpg"
                        new_path = os.path.join(class_dir, new_name)
                        rgb_img.save(new_path, "JPEG")
                        os.remove(img_path)

            except Exception as e:
                print(f"[⚠️] Error processing {img_path}: {e}")
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"[⚠️] Failed to remove {img_path}: {e}")

def augment_image(img):
    transformations = []
    # Поворот
    angle = random.choice([90, 180, 270])
    transformations.append(img.rotate(angle))
    # Зеркало
    transformations.append(ImageOps.mirror(img))
    # Яркость
    enhancer = ImageEnhance.Brightness(img)
    transformations.append(enhancer.enhance(random.uniform(0.6, 1.4)))
    # Контраст
    contrast = ImageEnhance.Contrast(img)
    transformations.append(contrast.enhance(random.uniform(0.6, 1.4)))
    # Цвет
    color = ImageEnhance.Color(img)
    transformations.append(color.enhance(random.uniform(0.5, 1.5)))
    return transformations

def augment_dataset():
    print("[🔁] Augmenting images...")
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"[⚠️] Directory {class_dir} does not exist, skipping augmentation.")
            continue

        images = [img for img in os.listdir(class_dir) if img.endswith(".jpg")]
        if len(images) == 0:
            print(f"[⚠️] No valid images found for class '{class_name}', skipping augmentation.")
            continue

        index = len(images)
        while len(images) < TARGET_COUNT:
            img_name = random.choice(images)
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    aug_images = augment_image(img)
                    for aug_img in aug_images:
                        if len(images) >= TARGET_COUNT:
                            break
                        aug_name = f"aug_{index}.jpg"
                        aug_path = os.path.join(class_dir, aug_name)
                        aug_img.save(aug_path)
                        images.append(aug_name)
                        index += 1
            except Exception as e:
                print(f"[⚠️] Error augmenting {img_path}: {e}")

def create_preview():
    print("[🖼] Creating preview HTML...")
    os.makedirs(os.path.dirname(PREVIEW_HTML), exist_ok=True)
    with open(PREVIEW_HTML, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>Dataset Preview</h1>")
        for class_name in CLASSES:
            class_dir = os.path.join(DATASET_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"[⚠️] Directory {class_dir} does not exist, skipping in preview.")
                continue

            f.write(f"<h2>{class_name.capitalize()}</h2><div style='display:flex;flex-wrap:wrap;'>")
            images = [img for img in os.listdir(class_dir) if img.endswith(".jpg")][:10]
            for img in images:
                rel_path = os.path.join("dataset", class_name, img).replace("\\", "/")
                f.write(f"<div style='margin:5px'><img src='{rel_path}' height='150'></div>")
            f.write("</div>")
        f.write("</body></html>")
    print(f"[✅] Preview saved to: {PREVIEW_HTML}")

if __name__ == "__main__":
    download_images()
    clean_images()
    augment_dataset()
    create_preview()