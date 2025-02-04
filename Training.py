# Import libraries
import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torchmetrics
import warnings
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import ResNet18
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# Logging evaluation metrics during training with TensorBoard and storing in "logs" folder
def log_model(epoch, avg_loss, accuracy, precision, recall):
    # 1: create TensorBoard writer by setting its path
    # 2: log evaluation metrics to TensorBoard

    # (1)
    log_dir = "./logs"
    writer = SummaryWriter(log_dir=log_dir)

    # (2)
    writer.add_scalar("Loss of training", avg_loss, epoch + 1)
    writer.add_scalar("Accuracy of training", accuracy, epoch + 1)
    writer.add_scalar("Precision of training", precision, epoch + 1)
    writer.add_scalar("Recall of training", recall, epoch + 1)


# Plotting the loss curves and other evaluation metrics with matplotlib and storing in "plots" folder
def plot_model(epochs_list, losses, accuracies, precisions, recalls, f1_scores):
    # 1: create plots directory
    # 2: define evaluation metrics that have been used during training
    # 3: draw the plots for the above-mentioned metrics
    # 4: save the plot into the created path no.1

    # (1)
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    # (2)
    metrics = {
        "Loss": losses,
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
    }

    # (3)
    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_list, values, marker="o", linestyle="-", label=metric_name, color="r")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over Epochs")
        plt.legend()
        plt.grid()

        # (4)
        plot_path = os.path.join(plot_dir, f"{metric_name}.png")
        plt.savefig(plot_path)
        plt.close()


# Main function to train the model on the cats vs. dogs binary classification task
def main():
    # 1: initialization -- define vectors, variables, loss and optimizer to be used in the training process
    # 2: we apply here an online preprocessing step; besides, there is an already preprocessed dataset
    # 3: set the model requirements

    # (1)
    epochs_list, losses, accuracies, precisions, recalls, f1_scores = [], [], [], [], [], []
    all_predict, all_labels =  [], []
    # We choose to have multiple epochs (10) that help the model learn better representations,
    # but not stuck in overfitting or underfitting.
    epochs = 10
    # Each epoch start with zero loss, then we calculate its loss during training
    running_loss = 0.0

    # (2)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    # ImageFolder: help load dataset structured into class folders and apply transformation, it works
    # well with DataLoader and do transformation efficiently
    # DataLoader:  help load, batch and shuffle ImageFolder, while avoiding loading everything once into memory
    # batch_size=32 is a hyperparameter balancing Memory Usage & Training Speed
    train_dataset = ImageFolder(root="Dataset/train", transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # (3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet18().to(device)

    # Create directory for saving model checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize variable to track the best loss meaning the lowest amount during training for each epoch
    # Since we want to find the smallest loss, starting with infinity (inf) ensures that any valid loss will be smaller
    best_loss       = float("inf")
    best_model_path = ""

    # (1)
    # We can apply specific loss function for binary classification purposes that handles Sigmoid activation automatically;
    # alternatively we can use CrossEntropyLoss function that is general for multi-class classification
    # We apply Adam (Adaptive Moment Estimation) that automatically adjusts the learning rate for each parameter
    # Learning rate = 0.001 is chosen to avoid making training unstable (e.g. lr = 0.01) or slow (e.g. lr = 0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4: start of the training process within 10 epochs which consists of
    # 4.1: train the model
    # 4.2: move Data and labels to the Device
    # 4.3: core backpropagation and optimization process in PyTorch
    # 4.4: calculation of loss to then get the average in each epoch
    # 4.5: find predicted class for a binary classification
    # 4.6: calculate the evaluation metrics on predictions and true labels
    # 4.7: store the evaluation metrics' result for further presentation

    # (4)
    for epoch in range(epochs):
        # (4.1)
        model.train()

        for images, labels in train_loader:
            # (4.2)
            images, labels = images.to(device), labels.to(device)

            # (4.3)
            # This step is important in Adam optimizer. Incrementally, the gradients are accumulated
            # after backward, forward passes to avoid gradient from previous batch affects current one.
            optimizer.zero_grad()
            # We perform the forward pass and defines how the input image is transformed into
            # the outputs that are logits or predicted values from the model.
            outputs = model(images)
            # After executing model on images, we got the output and to compare with true labels, we use the previously
            # defined loss function.
            loss = criterion(outputs, labels)
            # Here after setting gradient to zero in optimizer.zero_grad(), we calculate the gradient newly
            # for the current batch
            loss.backward()
            # Now by having the gradients computed in loss.backward(); we update the model weights
            optimizer.step()

            # (4.4)
            running_loss += loss.item()

            # (4.5)
            _, preds = torch.max(outputs, 1)
            all_predict.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # (4.6)
        accuracy  = accuracy_score(all_labels, all_predict)
        precision = precision_score(all_labels, all_predict, average="binary")
        recall    = recall_score(all_labels, all_predict, average="binary")
        avg_loss  = running_loss / len(train_loader)


        # (4.7)
        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        # call log model to log the evaluation metrics during training
        log_model(epoch, avg_loss, accuracy, precision, recall)

        # call plot model to provide plots of loss curves and other metrics in a dedicated folder
        plot_model(epochs_list, losses, accuracies, precisions, recalls, f1_scores)

        # 5: Automatically save the best model checkpoint and print its absolute path which consists of
        # 5.1: check if the loss has improved
        # 5.2: reset the best loss if applicable
        # 5.3: save the best model

        # (5)
        if avg_loss < best_loss:
            # (5.1)
            best_loss = avg_loss
            # (5.2)
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            # (5.3)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at: {os.path.abspath(best_model_path)}")

    print("Training is successfully completed.")


if __name__ == '__main__':
    main()



