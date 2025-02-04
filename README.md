# Image-Classification
In this project, we implement an **image classification model** to distinguish between two categories of images (**cats
vs. dogs**) in **PyTorch**. Then, we **train** the model, **evaluate** and **interpret** the results.

We develop based on the [ResNet model](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py); while configuring it to match out preprocessed input and the number of classes. After training model on train dataset, we log and plot the results for each epoch. Finally, we test the model on a single image in the test dataset to see the predicted class as well as the confidence score.

## Installation
First you need to ensure that all packages have been installed.
+ See `requirements.txt`
+ torch>=2.4.1
+ torchvision>=0.19.1
+ matplotlib>=3.7.1
+ scikit-learn>=1.1.1
+ Pillow>=9.2.0

You can clone this repository:

	   > git clone https://github.com/ShokofehVS/Image-Classification.git

If you miss something you can simply type:

	   > pip install -r requirements.txt

If you have all dependencies installed:

	   > pip3 install .

## Dataset
[Animal Classification Dataset (Cats, Dogs, Horses)](https://www.kaggle.com/datasets/arifmia/animal/data) involves three common animal categories: Cat, Dog, and Horse.  It includes a total of 1,763 images split into three subsets: Train, Validation (Val), and Test. 

# Image Classification Steps:
## Step 1: Preprocessing
After accommodating an extracted dataset into a **Dataset** folder, we remove **horse** category; this can be done by writing a code in which define exclusively two categories of **dogs** and **cats**:

Further, we transform the training dataset composing of images of cats and dogs into 64x64 pixels and grayscale (1 channel):

`transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])`

The script for running preprocessing is `Preprocessing.py`. However, we devoted a space called `Preprocessed_animal_data` for this offline preprocessing on training datasets.

Despite an offline preprocessing, we additionally can preprocess online in ``Training.py`` before starting training. 

## Step 2: Training
