import os
import random
import shutil

source_dir = r"C:\Users\yasha\Desktop\Annamitra project\dataset_flat"
target_dir = r"C:\Users\yasha\Desktop\Annamitra project\dataset"

split_ratio = (0.7, 0.2, 0.1)  # train, validation, test

for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue

    for food_class in os.listdir(category_path):
        class_path = os.path.join(category_path, food_class)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        train_count = int(len(images) * split_ratio[0])
        val_count = int(len(images) * split_ratio[1])

        splits = {
            "train": images[:train_count],
            "validation": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:]
        }

        for split_name, split_images in splits.items():
            split_folder = os.path.join(target_dir, split_name, food_class)
            os.makedirs(split_folder, exist_ok=True)

            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_folder, img)
                shutil.copy(src, dst)

print("Dataset split completed successfully!")