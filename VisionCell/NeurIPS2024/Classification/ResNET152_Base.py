import os
import re
import sys
import copy
import torch
import logging
import argparse
import torchvision
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def get_input():
    parser = argparse.ArgumentParser(description='Training ResNET152 for classification of Cell Images.')
    parser.add_argument('--learning_rate',     type=float, default=0.01,               help='Learning rate')
    parser.add_argument('--momentum',          type=float, default=0.9,                help='Momentum')
    parser.add_argument('--num_epochs',        type=int,   default=2,                  help='Number of epochs')
    parser.add_argument('--patience',          type=int,   default=5,                  help='Patience for early stopping')
    parser.add_argument('--min_improvement',   type=float, default=0.1,                help='Minimum improvement to reset patience counter')
    parser.add_argument('--image_resolution',  type=str,   default='224,224',          help='Image resolution as width,height')
    parser.add_argument('--step_size',         type=int,   default=5,                  help='Step size for learning rate decay')
    parser.add_argument('--gamma',             type=float, default=0.1,                help='Decay rate (gamma) for learning rate')
    parser.add_argument('--loss_function',     type=str,   default='CrossEntropyLoss', help='Loss function choice')
    parser.add_argument('--data_dir',          type=str,   required=True,              help='Directory path for the data')
    parser.add_argument('--split_ratio',       type=float, default=0.8,                help='Ratio of data for training. The rest is for validation.')
    parser.add_argument('--batch_size',        type=int,   default=16,                 help='Batch size for training and validation')
    parser.add_argument('--num_classes',       type=int,   default=2,                  help='Number of classes for classification')
    args = parser.parse_args()
    return args

def initialize_loss_function(choice):
    if choice == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif choice == 'MultiMarginLoss':
        return nn.MultiMarginLoss()
    elif choice == 'KLDivLoss':
        return nn.KLDivLoss(reduction='batchmean')
    elif choice == 'NLLLoss':
        return nn.NLLLoss()
    elif choice == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif choice == 'SoftMarginLoss':
        return nn.SoftMarginLoss()
    else:
        logging.error("Invalid loss function choice")
        sys.exit("Invalid loss function choice")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_label = {classname: index for index, classname in enumerate(self.classes)}
        self.image_paths = [os.path.join(root_dir, classname, filename)
                    for classname in self.classes
                    for filename in os.listdir(os.path.join(root_dir, classname))
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Class to Label Mapping: {self.class_to_label}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if image.size[0] < 224 or image.size[1] < 224:
            raise ValueError(f"Image at {img_path} has size {image.size}, which is below the required size of (224, 224).")
        
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_label[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define the bottleneck block
def bottleneck_block(in_channels, out_channels, stride=1, downsample=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels * 4),
        nn.ReLU(inplace=True),
        downsample
    )

# Define the CustomResNet152
class CustomResNet152(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomResNet152, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(256, 36, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers = []
        layers.append(bottleneck_block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(bottleneck_block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Checking the architecture
model = CustomResNet152()
print(model)

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    labels       = labels.cpu().numpy()
    predicted    = predicted.cpu().numpy()
    precision    = precision_score(labels, predicted, average='weighted', zero_division=1)
    recall       = recall_score(labels, predicted, average='weighted', zero_division=1)
    f1_metric    = f1_score(labels, predicted, average='weighted', zero_division=1)
    accuracy     = np.mean(labels == predicted)
    return accuracy, precision, recall, f1_metric

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
    
def plot_loss(epochs, train_losses, val_losses, metric_name, x_label, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.plot(epochs, val_losses,   'b', label='Validation Loss')
    plt.title(f'{x_label} vs {metric_name}')
    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_all_metrics(epochs, accuracy_list, f1_list, specificity_list, precision_list, recall_list, title, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, accuracy_list,    label='Accuracy', linestyle='-')
    plt.plot(epochs, f1_list,          label='F1-Score', linestyle='--')
    plt.plot(epochs, specificity_list, label='Specificity', linestyle='-.')
    plt.plot(epochs, precision_list,   label='Precision', linestyle=':')
    plt.plot(epochs, recall_list,      label='Recall', linestyle='-.')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_metrics_vs_f1(metric_values, f1_values, metric_name, title, filename):
    plt.figure()
    plt.plot(metric_values, f1_values, '-o', label=f'{metric_name} vs F1-Score')
    plt.title(title)
    plt.xlabel(metric_name)
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('ResNET152_Cell_confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    args = get_input()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA-enabled GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        
    learning_rate        = args.learning_rate
    momentum             = args.momentum
    num_epochs           = args.num_epochs
    patience             = args.patience
    min_improvement      = args.min_improvement
    image_resolution     = tuple(map(int, args.image_resolution.split(',')))
    data_dir             = args.data_dir
    batch_size           = args.batch_size
    num_classes          = args.num_classes
    split_ratio          = args.split_ratio
    loss_function_choice = args.loss_function
    
    transform = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_dataset = CustomDataset(root_dir=data_dir, transform=transform)
    train_size = int(split_ratio * len(all_dataset))
    val_size = len(all_dataset) - train_size
    
    if val_size == 0:
        sys.exit("Validation dataset size is zero. Please check your data split ratio.")
    
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = torchvision.models.resnet152(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    
    criterion = initialize_loss_function(loss_function_choice)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    accuracy_list, precision_list, recall_list, f1_list, specificity_list, train_loss_list, val_loss_list = [], [], [], [], [], [], []
        
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    
    log_filename = f"ResNET152_Cell_log_{run_timestamp}.txt"

    with open(log_filename, 'w') as log_file:
        log_file.write("Epoch\tTraining Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tF1-Score\tSpecificity\n")

    for epoch in range(num_epochs):
    
        try:
            print(f"Starting Epoch {epoch + 1}/{num_epochs}...")

            running_train_loss = 0.0
            running_val_loss   = 0.0
            val_acc            = 0.0
            val_precision      = 0.0
            val_recall         = 0.0
            val_f1             = 0.0
            val_specificity    = 0.0
            
            # Training Phase
            model.train()
            num_train_batches = len(trainloader)
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if torch.isnan(outputs).any():
                    print("NaN values detected in model outputs. Replacing with default values.")
                    outputs[torch.isnan(outputs)] = 0.0
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss).item():
                    print("NaN values detected in loss. Skipping this batch.")
                    continue
                    
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / num_train_batches

            # Validation Phase
            model.eval()
            num_val_batches = len(valloader)
            all_labels, all_predictions = [], []

            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    if torch.isnan(outputs).any():
                        print("NaN values detected in model outputs during validation. Replacing with default values.")
                        outputs[torch.isnan(outputs)] = 0.0

                    loss = criterion(outputs, labels)

                    if torch.isnan(loss).item():
                        print("NaN values detected in validation loss. Skipping this batch.")
                        continue

                    running_val_loss += loss.item()

                    acc, prec, rec, f1_val = calculate_metrics(outputs, labels)
                    val_acc += acc
                    val_precision += prec
                    val_recall += rec
                    val_f1 += f1_val
                    
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            avg_val_loss = running_val_loss / num_val_batches
            val_acc /= num_val_batches
            val_precision /= num_val_batches
            val_recall /= num_val_batches
            val_f1 /= num_val_batches
            val_specificity = specificity_score(all_labels, all_predictions)

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
            print(f"Accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1-Score: {val_f1}, Specificity: {val_specificity}")
            
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
            accuracy_list.append(val_acc)
            precision_list.append(val_precision)
            recall_list.append(val_recall)
            f1_list.append(val_f1)
            specificity_list.append(val_specificity)

            # Log the metrics to the log file
            with open(log_filename, 'a') as log_file:
                log_file.write(f"{epoch + 1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc:.4f}\t{val_precision:.4f}\t{val_recall:.4f}\t{val_f1:.4f}\t{val_specificity:.4f}\n")

            # Early stopping logic
            if avg_val_loss < best_loss - min_improvement:
                best_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Early stopping after {epoch + 1} epochs with no improvement.")
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"Early stopping after {epoch + 1} epochs with no improvement.\n")
                break

        except Exception as e:
            error_msg = f"Error during Epoch {epoch + 1}: {str(e)}"
            print(error_msg)
            
            with open(log_filename, 'a') as log_file:
                log_file.write(error_msg + "\n")
            break

    model.load_state_dict(best_model_weights)
    best_model_name = f"trained_resnet152_Cell_bestmodel_{run_timestamp}.pth"
    torch.save(model.state_dict(), best_model_name)
    
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    class_names = list(all_dataset.class_to_label.keys())

    classification_rep = classification_report(all_labels, all_predictions, target_names=class_names)
    print(classification_rep)
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, class_names)

    epochs = range(1, len(train_loss_list) + 1)
    plot_loss(epochs, train_loss_list, val_loss_list, 'Loss', 'Epoch', f'ResNET152_Cell_training_validation_loss_{run_timestamp}.png')

    plot_all_metrics(epochs, accuracy_list, f1_list, specificity_list, precision_list, recall_list, "Performance Over Epochs", f'ResNET152_Cell_all_metrics_{run_timestamp}.png')
        
    metrics_for_comparison = [accuracy_list, recall_list, precision_list, specificity_list]
    metric_names_for_comparison = ['Accuracy', 'Recall', 'Precision', 'Specificity']

    for metric, metric_name in zip(metrics_for_comparison, metric_names_for_comparison):
        plot_metrics_vs_f1(metric, f1_list, metric_name,
                           f'{metric_name} vs F1-Score', f'ResNET152_Cell_{metric_name}_Vs_F1-Score_{run_timestamp}.png')
        
