# Image-Classification
In this project, we implement an **image classification model** to distinguish between two categories of images (**cats
vs. dogs**) in **PyTorch**. Then, we **train** the model, **evaluate** and **interpret** the results.

We rely on ResNet model based on the [ResNet model](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py); while configuring it to match out preprocessed input and the number of classes. After training model on train dataset, we log and plot the results for each epoch. Finally, we test the model on a single image in the test dataset to see the predicted class as well as the confidence score.

## Installation
First you need to ensure that all packages have been installed.
+ See `requirements.txt`
+ torch>=2.4.1
+ torchvision>=0.19.1
+ matplotlib>=3.7.1
+ scikit-learn>=1.1.1
+ Pillow>=9.2.0
+ torchmetrics>=1.5.2

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

To run the offline preprocessing step:

	   > python3 Preprocessing.py

Despite an offline preprocessing, we additionally can preprocess online in ``Training.py`` before starting training. 

## Step 2: Training
We rely on ResNet model based on the [ResNet model](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py); while configuring it to match out preprocessed input (e.i., grayscale images (1 channel)) and the number of classes (cats and dogs, 2). We only keep the ResNet18 because it has fewer parameters, making it faster to train; 
especially useful when there are limited compute resources. 


Here we explain the model in details for better clarity:

**ResNet (Residual Network)** is based on the idea of residual learning and is composed of residual blocks. In the above-mentioned implementation:

- Basic Blocks: it contains of two convolutional layers followed by batch normalization and ReLU activation. 
- ResNet: it defines the overall architecture of the model, using multiple residual blocks and applying them in a sequential manner.

To run the training phase:

	   > python3 Training.py

## Step 3: Testing
Finally, we evaluate the model on the test dataset, where we extract one sample image in the test directory for category of **dogs** and find the prediction result as well as the confidence score.

To run the testing phase:

	   > python3 Testing.py

To sum up, we observe that with confidence 0.5527, a true label **dog** is set for an input in the testing dataset.

## Challenges:
Throughout the implementation, we notice some challenges especially in dealing with transformed images in model training. We configure the model to accept 1 channel image with 64x64 pixels.

## Future Directions:
In our model training, we set hyperpyrameters than can be improved experimentally and applying larger set of data:

Learning rate (lr)        = 0.001

Number of epochs (epochs) = 10

batch size                = 64


We run the experiment on an Intel(R) Core i7-1185 CPU at 3.00 GHz × 8 running Ubuntu 20.04 LTS with 31 GB memory.