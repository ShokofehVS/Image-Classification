# Import libraries
import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import ResNet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter



# Logging evaluation metrics during training with TensorBoard and storing in "logs" folder
def log_model(epoch, avg_loss, accuracy, precision, recall, f1):
    # 1: create TensorBoard writer by setting its path
    # 2: log evaluation metrics to TensorBoard

    # (1)
    log_dir = "./logs"
    writer = SummaryWriter(log_dir=log_dir)

    # (2)
    writer.add_scalar("Loss/train", avg_loss, epoch + 1)
    writer.add_scalar("Accuracy/train", accuracy, epoch + 1)
    writer.add_scalar("Precision/train", precision, epoch + 1)
    writer.add_scalar("Recall/train", recall, epoch + 1)
    writer.add_scalar("F1-Score/train", f1, epoch + 1)


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
        "F1-Score": f1_scores
    }

    # (3)
    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_list, values, marker="o", linestyle="-", label=metric_name, color="b")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over Epochs")
        plt.legend()
        plt.grid()

        # (4)
        plot_path = os.path.join(plot_dir, f"{metric_name.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")


# Main function to train the model on the cats vs. dogs binary classification task
def main():
    # 1: initialization -- define vectors, variables, loss and optimizer to be used in the training process
    # 2: we apply here an online preprocessing step; besides, there is an already preprocessed dataset
    # 3: set the model requirements

    # (1)
    epochs_list, losses, accuracies, precisions, recalls, f1_scores = [], [], [], [], [], []
    all_preds, all_labels =  [], []
    epochs = 10
    running_loss = 0.0

    # (2)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root="Dataset/train", transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # (3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet18().to(device)

    # Create directory for saving model checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize variable to track the best loss
    best_loss = float("inf")
    best_model_path = ""

    # (1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4: start of the training process within 10 epochs which consists of
    # 4.1: train the model
    # 4.2: move Data to the Device
    # 4.3: core backpropagation and optimization process in PyTorch
    # 4.4: calculation of loss to then get the average in each epoch
    # 4.5: find predicted class with highest logit value
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
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # (4.4)
            running_loss += loss.item()

            # (4.5)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # (4.6)
        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="binary")
        recall    = recall_score(all_labels, all_preds, average="binary")
        f1        = f1_score(all_labels, all_preds, average="binary")
        avg_loss  = running_loss / len(train_loader)

        # (4.7)
        epochs_list.append(epoch + 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # call log model to log the evaluation metrics during training
        log_model(epoch, avg_loss, accuracy, precision, recall, f1)

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



