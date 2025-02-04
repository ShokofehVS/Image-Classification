# Import libraries
import os
import torchvision.transforms as transforms
import shutil
from PIL import Image


def main(data_dir, preprocessed_data_dir):
    # Preprocessing function that
    # 1. Makes directory to store preprocessed images
    # 2: Categorizes the dataset into two cats and dogs (ignoring horses)
    # 3. Ensures input images are 64x64 pixels
    # 3. Converts images to grayscale (1 channel)
    # 4: Saves final results into the defined directory (refer to no. 1)

    # (1)
    os.makedirs(preprocessed_data_dir, exist_ok=True)

    # (2)
    categories = ["cat", "dog"]
    for category in categories:
        os.makedirs(os.path.join(preprocessed_data_dir, category), exist_ok=True)

    #  (3)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1)
    ])

    #  (4)
    for category in categories:
        # in our Preprocessed dataset (dest_path) we have exact category like in the train dataset (src_path):
        # os.path.join: concatenate multiple paths such as data_dir, preprocessed_data_dir with suitable category
        # os.listdir:   list all files and directories in a specific directory, here src_path to go through all images
        # Image.open:   open an image files derived from Pillow library, here img_path for specific category
        # transform:    transformation of the source image into 64x64 pixels, and grayscale (1 channel)
        # save:         save the preprocessed image into its path, here dest_path
        src_path  = os.path.join(data_dir, category)
        dest_path = os.path.join(preprocessed_data_dir, category)

        for img_name in os.listdir(src_path):
            img_path = os.path.join(src_path, img_name)
            try:
                img  = Image.open(img_path)
                img  = transform(img)
                img.save(os.path.join(dest_path, img_name))
            except Exception as e:
                print(f"Error processing     {img_path}: {e}")

    print("Preprocessing is completed. Images are stored in", preprocessed_data_dir)


if __name__ == '__main__':
    # Path to the file where datasets after extraction and preprocessing respectively exist
    data_dir              = "./Dataset/train"
    preprocessed_data_dir = "./Preprocessed_animal_data/train"

    # Run the main function to preprocess the training dataset
    main(data_dir, preprocessed_data_dir)
