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
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def get_input():
    parser = argparse.ArgumentParser(description='Training Inception V3 for classification of Cell Images.')
    parser.add_argument('--learning_rate',     type=float, default=0.01,               help='Learning rate')
    parser.add_argument('--momentum',          type=float, default=0.9,                help='Momentum')
    parser.add_argument('--num_epochs',        type=int,   default=2,                  help='Number of epochs')
    parser.add_argument('--patience',          type=int,   default=5,                  help='Patience for early stopping')
    parser.add_argument('--min_improvement',   type=float, default=0.1,                help='Minimum improvement to reset patience counter')
    parser.add_argument('--image_resolution',  type=str,   default='299,299',          help='Image resolution as width,height')
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
        
        if image.size[0] < 299 or image.size[1] < 299:
            raise ValueError(f"Image at {img_path} has size {image.size}, which is below the required size of (224, 224).")
        
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_label[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class InceptionV3Attention(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Attention, self).__init__()
        # Load the Inception v3 model, make sure to set aux_logits to False
        self.inception_v3 = models.inception_v3(pretrained=False, aux_logits=False)
        # Remove the last average pooling and the fully connected layer
        # This assumes that the last two layers of features are these components.
        # If the architecture is different, adjust the indices accordingly.
        self.inception_v3 = nn.Sequential(*list(self.inception_v3.children())[:-2])
        # Add the SEBlock
        self.se_block = SEBlock(2048)  # Assuming the last channel dimension is 2048
        # Add the final classification layer
        self.final_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Get the features from the inception model
        x = self.inception_v3(x)
        # Apply the SEBlock
        x = self.se_block(x)
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # Classification layer
        x = self.final_fc(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"SEBlock forward method expects a 4D tensor but got a tensor with {x.ndim} dimensions")
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    labels       = labels.cpu().numpy()
    predicted    = predicted.cpu().numpy()
    precision    = precision_score(labels, predicted, average='weighted', zero_division=1)
    recall       = recall_score(labels,    predicted, average='weighted', zero_division=1)
    f1_metric    = f1_score(labels,        predicted, average='weighted', zero_division=1)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('InceptionV3_Attention_Confusion_Matrix.png')
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
    
    # Instantiate the InceptionV3Attention model
    model = InceptionV3Attention(num_classes=num_classes)
    model = model.to(device)
    
    criterion = initialize_loss_function(loss_function_choice)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    accuracy_list, precision_list, recall_list, f1_list, specificity_list, train_loss_list, val_loss_list = [], [], [], [], [], [], []
        
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    
    log_filename = f"InceptionV3_Attention_log_{run_timestamp}.txt"

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
                #loss_main = criterion(outputs[0], labels)
                #loss_aux = criterion(outputs[1], labels)
                #loss = loss_main + 0.4 * loss_aux
                loss = criterion(outputs, labels)
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
                    loss = criterion(outputs, labels)

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
    best_model_name = f"InceptionV3_Attention_trained_bestmodel_{run_timestamp}.pth"
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
    plot_loss(epochs, train_loss_list, val_loss_list, 'Loss', 'Epoch', f'InceptionV3_Attention_training_validation_loss_{run_timestamp}.png')

    plot_all_metrics(epochs, accuracy_list, f1_list, specificity_list, precision_list, recall_list, "Performance Over Epochs", f'InceptionV3_Attention_all_metrics_{run_timestamp}.png')
        
    metrics_for_comparison = [accuracy_list, recall_list, precision_list, specificity_list]
    metric_names_for_comparison = ['Accuracy', 'Recall', 'Precision', 'Specificity']

    for metric, metric_name in zip(metrics_for_comparison, metric_names_for_comparison):
        plot_metrics_vs_f1(metric, f1_list, metric_name,
                           f'{metric_name} vs F1-Score', f'InceptionV3_Attention_{metric_name}_Vs_F1-Score_{run_timestamp}.png')
        
