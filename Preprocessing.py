# Import libraries
import os
import torchvision.transforms as transforms
from PIL import Image
import shutil

# Path to the file where datasets after extraction and preprocessing respectively exist
data_dir = "./Dataset/train"
preprocessed_data_dir = "Preprocessed_animal_data/train"

# Preprocessing function that
# 1. Makes directory to store preprocessed images
# 2: Categorizes the dataset into two cats and dogs
# 2. Ensures input images are 64x64 pixels
# 3. Converts images to grayscale (1 channel)
# 4. Additional preprocessing steps
# 5: Saves final results into the defined directory (refer to no. 1)

# (1)
if os.path.exists(preprocessed_data_dir):
    shutil.rmtree(preprocessed_data_dir)
os.makedirs(preprocessed_data_dir, exist_ok=True)

# (2)
categories = ["cat", "dog"]
for category in categories:
    os.makedirs(os.path.join(preprocessed_data_dir, category), exist_ok=True)

#  (3, 4)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1)
])

#  (5)
for category in categories:
    src_path  = os.path.join(data_dir, category)
    dest_path = os.path.join(preprocessed_data_dir, category)

    if not os.path.exists(src_path):
        print(f"Warning: Source directory {src_path} does not exist.")
        continue

    for img_name in os.listdir(src_path):
        img_path = os.path.join(src_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            img.save(os.path.join(dest_path, img_name))
        except Exception as e:
            print(f"Error processing     {img_path}: {e}")

print("Preprocessing complete. Images are stored in", preprocessed_data_dir)

