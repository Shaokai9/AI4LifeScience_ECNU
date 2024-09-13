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
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def get_input():
    parser = argparse.ArgumentParser(description='Training Inception V3 for classification of Curve Images.')
    parser.add_argument('--learning_rate',     type=float, default=0.01,               help='Learning rate')
    parser.add_argument('--momentum',          type=float, default=0.9,                help='Momentum')
    parser.add_argument('--num_epochs',        type=int,   default=5,                  help='Number of epochs')
    parser.add_argument('--patience',          type=int,   default=10,                 help='Patience for early stopping')
    parser.add_argument('--min_improvement',   type=float, default=0.1,                help='Minimum improvement to reset patience counter')
    parser.add_argument('--image_resolution',  type=str,   default='299,299',          help='Image resolution as width,height')
    parser.add_argument('--step_size',         type=int,   default=2,                  help='Step size for learning rate decay')
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
    plt.savefig('InceptionV3_Base_Confusion_Matrix.png')
    plt.show()

def setup_logging():
    logging.basicConfig(filename='training_log.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    args = get_input()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'CUDA-enabled GPU' if device.type == 'cuda' else 'CPU'}")

    learning_rate = args.learning_rate
    momentum = args.momentum
    num_epochs = args.num_epochs
    patience = args.patience
    min_improvement = args.min_improvement
    image_resolution = tuple(map(int, args.image_resolution.split(',')))
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_classes = args.num_classes
    loss_function_choice = args.loss_function

    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_resolution),
        transforms.RandomRotation(degrees=90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_dataset = CustomDataset(root_dir=data_dir, transform=transform)

    k = 5  # Number of folds for K-Fold Cross-Validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_performance = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(all_dataset)))):
        print(f'Fold {fold + 1}/{k}')

        train_subset = torch.utils.data.Subset(all_dataset, train_ids)
        val_subset = torch.utils.data.Subset(all_dataset, val_ids)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = models.inception_v3(pretrained=False, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = initialize_loss_function(loss_function_choice)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        best_loss = np.inf
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        fold_metrics = {'train_loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}

        for epoch in range(num_epochs):
            try:
                model.train()
                running_train_loss = 0.0
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):  # When the model returns both main and aux outputs
                        loss_main = criterion(outputs[0], labels)
                        loss_aux = criterion(outputs[1], labels)
                        loss = loss_main + 0.4 * loss_aux
                    else:  # When the model returns only the main output
                        loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                avg_train_loss = running_train_loss / len(trainloader)
                fold_metrics['train_loss'].append(avg_train_loss)

                model.aux_logits = False  # Set aux_logits to False for evaluation
                model.eval()
                all_labels, all_predictions = [], []
                running_val_loss = 0.0  # Initialize running validation loss

                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)  # Only the main output is returned
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()  # Accumulate the validation loss

                        _, predicted = torch.max(outputs, 1)
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                avg_val_loss = running_val_loss / len(valloader)  # Calculate average validation loss

                val_acc, val_precision, val_recall, val_f1 = calculate_metrics(all_predictions, all_labels)
                val_specificity = specificity_score(all_labels, all_predictions)

                fold_metrics['val_loss'].append(avg_val_loss)
                fold_metrics['accuracy'].append(val_acc)
                fold_metrics['precision'].append(val_precision)
                fold_metrics['recall'].append(val_recall)
                fold_metrics['f1'].append(val_f1)
                fold_metrics['specificity'].append(val_specificity)

                if avg_val_loss < best_loss - min_improvement:
                    best_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at fold {fold + 1}, epoch {epoch + 1}")
                    break

                scheduler.step()

            except Exception as e:
                print(f"Error during Epoch {epoch + 1}: {str(e)}")
                break

        model.load_state_dict(best_model_weights)
        fold_model_name = f"trained_InceptionV3_Fold{fold + 1}_bestmodel_{run_timestamp}.pth"
        torch.save(model.state_dict(), fold_model_name)

        fold_performance.append(fold_metrics)

    # Overall evaluation after all folds
    # Aggregating fold metrics
    aggregated_metrics = {metric: np.mean([fold[metric] for fold in fold_performance]) for metric in fold_performance[0]}

    # Printing classification report and confusion matrix for the last fold
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

    # Plotting metrics
    epochs = range(1, num_epochs + 1)
    plot_loss(epochs, aggregated_metrics['train_loss'], aggregated_metrics['val_loss'], 'Loss', 'Epoch', f'InceptionV3_Base_training_validation_loss_{run_timestamp}.png')
    plot_all_metrics(epochs, aggregated_metrics['accuracy'], aggregated_metrics['f1'], aggregated_metrics['specificity'], aggregated_metrics['precision'], aggregated_metrics['recall'], "Performance Over Epochs", f'InceptionV3_Base_all_metrics_{run_timestamp}.png')

    for metric, values in aggregated_metrics.items():
        if metric not in ['train_loss', 'val_loss']:
            plot_metrics_vs_f1(values, aggregated_metrics['f1'], metric, f'{metric} vs F1-Score', f'InceptionV3_Base_{metric}_Vs_F1-Score_{run_timestamp}.png')
