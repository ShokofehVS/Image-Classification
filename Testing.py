# Import libraries
import torch
import torchvision.transforms as transforms
import warnings
from models import ResNet18
from PIL import Image

warnings.filterwarnings("ignore")

# Load the trained model from checkpoint
def load_model(checkpoint_path):
    # 1: move data/ model to device
    # 2: initialize the same model architecture in training here in testing
    # 3: copy the saved parameters into the new model
    # 4: set model to evaluation mode
    # (1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # (2)
    model = ResNet18()
    # (3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # (1)
    model.to(device)
    # (4)
    model.eval()

    return model, device


# Predict class label of a single image
def predict_image(transform, image_path, model, device):
    # 1: binary classification with two labels
    # 2: open the single image in the testing dataset
    # 3: apply the same preprocessing as in the training, making 3D input to 4D
    # 4: apply the model on the transformed image
    # 5: convert logits to probabilities
    # 6: find the highest probability to get the confidence score as well as predication class label
    # (1)
    CLASS_LABELS = {0: "Cat", 1: "Dog"}
    # (2)
    image = Image.open(image_path)
    # (3)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        # (4)
        output = model(image)
        # (5)
        probabilities = torch.softmax(output, dim=1)
        # (6)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = CLASS_LABELS[predicted_class.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score


# Main function to test the model on the cats vs. dogs binary classification task
def main():
    # 1: set the input image in the testing dataset, and the model checkpoint pathes
    # 2: we apply here an online preprocessing step similar to the one in the training phase
    # 3: load a trained model from a checkpoint
    # 4: predict the class ("Cat" or "Dog")
    # 5: print the predicted class ("Cat" or "Dog") and confidence score

    # (1)
    test_image_path = "./Dataset/test/dog/0d384a19cc29aaf56b4b890b79fbdc91423ccd9b11e3cd9a3d490262dfe0c658d768c6c_1920.jpg"
    checkpoint_path = "./checkpoints/best_model.pth"

    # (2)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # (3)
    model, device = load_model(checkpoint_path)

    # (4)
    predicted_class, confidence = predict_image(transform, test_image_path, model, device)

    # (5)
    print(f"Predicted Class: {predicted_class}, Confidence Score: {confidence:.4f}")


if __name__ == '__main__':
    main()
