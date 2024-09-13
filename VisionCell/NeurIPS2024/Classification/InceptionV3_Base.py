import os # Imports the OS module, which provides functions for interacting with the operating system.
import re # Imports the regular expressions module for string searching and manipulation.
import sys # Imports the system-specific parameters and functions module.
import copy # Imports the copy module for shallow and deep copy operations.
import torch # Imports PyTorch, a popular deep learning library.
import logging # Imports the logging module, used for tracking events that occur when software runs.
import argparse # Imports the argparse module, used for writing user-friendly command-line interfaces.
import torchvision # Imports Torchvision, a package consisting of popular datasets, model architectures, and common image transformations for computer vision.
import numpy as np # Imports NumPy, a fundamental package for scientific computing in Python.
import seaborn as sns # Imports Seaborn, a Python visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.
from PIL import Image # Imports the Image class from the Python Imaging Library (PIL), used for opening, manipulating, and saving many different image file formats.
import torch.nn as nn # Imports PyTorch's neural network module.
import torch.optim as optim # Imports PyTorch's optimization module, which includes various optimization algorithms.
from datetime import datetime # Imports the datetime class from the datetime module, used for manipulating dates and times.
import matplotlib.pyplot as plt # Imports the pyplot interface from the Matplotlib library for plotting graphs.
import torch.nn.functional as F # Imports PyTorch's functional interface, which contains typical operations used for building neural networks like loss functions, activation functions, etc.
import torchvision.models as models # Imports model architectures from Torchvision.
import torchvision.transforms as transforms # Imports the transforms module from Torchvision, which provides common image transformations.
from torch.optim.lr_scheduler import StepLR # Imports the StepLR class, a learning rate scheduler that decays the learning rate of each parameter group by gamma every step_size epochs.
from torchvision.datasets import ImageFolder # Imports the ImageFolder class, a generic data loader where the images are arranged in this way: root/class_x/xxx.ext, root/class_y/yyy.ext, etc.
from torch.utils.data import DataLoader, Dataset # Imports DataLoader and Dataset classes from PyTorch, which are used for loading and handling datasets.
from sklearn.model_selection import train_test_split # Imports the train_test_split function for splitting arrays or matrices into random train and test subsets.
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score # Imports various functions for calculating classification metrics.

def get_input(): #Starts the definition of a function named get_input. Inside this function, an argparse.ArgumentParser object is created. This object is used to parse command-line arguments. Several arguments are added to the parser, each of which configures a different aspect of the training process for the Inception V3 model. This section of the code is crucial for setting up the model's training configuration, allowing for flexible adjustments through command-line arguments. The arguments cover various aspects of the training process, such as learning parameters, data handling, and model specifics.
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
    args = parser.parse_args() #Parses the command-line arguments and stores them in args.
    return args #Returns the parsed arguments for use elsewhere in the script.

# This function is designed to provide flexibility in choosing the loss function for the training process, allowing the user to specify the desired loss function through command-line arguments. It helps to adapt the training process to different kinds of classification problems.
def initialize_loss_function(choice): # Begins the definition of a function that initializes the loss function based on a given choice.
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
        logging.error("Invalid loss function choice") # Logs an error message if an invalid choice is passed.
        sys.exit("Invalid loss function choice") # Exits the program with a message indicating that an invalid loss function choice was made.

class CustomDataset(Dataset): # Begins the definition of a custom dataset class that extends PyTorch's Dataset class. This class is a crucial component for handling image datasets in PyTorch. It organizes the data, facilitates the retrieval of images and their corresponding labels, and applies any necessary transformations to the images.
    def __init__(self, root_dir, transform=None): # The constructor of the CustomDataset class. It initializes the dataset with a root directory and an optional transform.
        self.root_dir = root_dir # Stores the root directory where the data is located.
        self.transform = transform # Stores the transformation to be applied to the images.
        self.classes = [d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d))] # Creates a list of classes (subdirectories) in the root directory.
        self.class_to_label = {classname: index for index, classname in enumerate(self.classes)} # Maps each class name to a numerical label.
        self.image_paths = [os.path.join(root_dir, classname, filename) # Constructs a list of image paths for all images in the dataset. It filters the images to include only those with specific file extensions (.png, .jpg, .jpeg).
                    for classname in self.classes
                    for filename in os.listdir(os.path.join(root_dir, classname))
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Class to Label Mapping: {self.class_to_label}") # Prints the mapping from class names to labels.

    def __len__(self): # Defines a method to return the number of images in the dataset.
        return len(self.image_paths) # Returns the total number of image paths.

    def __getitem__(self, idx):# Defines a method to retrieve an image and its label by index.
        img_path = self.image_paths[idx] # Gets the path of the image at the specified index.
        image = Image.open(img_path).convert('RGB') # Opens the image at the given path and converts it to RGB format.
        
        if image.size[0] < 299 or image.size[1] < 299: # Checks if the image size is less than the required minimum (299x299 for Inception V3).
            raise ValueError(f"Image at {img_path} has size {image.size}, which is below the required size of (224, 224).") # Raises a ValueError if the image size is below the required size.
        
        label_name = os.path.basename(os.path.dirname(img_path)) # Extracts the label name from the image path.
        label = self.class_to_label[label_name] # Converts the label name to a numerical label.
        
        if self.transform: # Checks if a transformation is to be applied to the image.
            image = self.transform(image) # Applies the transformation to the image.
            
        return image, label # Returns the transformed image and its label.
        
def calculate_metrics(outputs, labels): # Defines a function for calculating various metrics. These sections of the code are critical for handling image data and evaluating the performance of the trained model. The CustomDataset class provides a way to load and preprocess images and their associated labels, while the calculate_metrics function computes various performance metrics to assess the effectiveness of the model.
    _, predicted = torch.max(outputs, 1) # Finds the predicted labels by selecting the class with the highest output value.
    labels       = labels.cpu().numpy()  # Converts the labels from PyTorch tensors to NumPy arrays.
    predicted    = predicted.cpu().numpy() # Converts the predicted labels to NumPy arrays.
    precision    = precision_score(labels, predicted, average='weighted', zero_division=1) # Calculates the weighted precision.
    recall       = recall_score(labels,    predicted, average='weighted', zero_division=1) # Calculates the weighted recall.
    f1_metric    = f1_score(labels,        predicted, average='weighted', zero_division=1) # Calculates the weighted F1 score.
    accuracy     = np.mean(labels == predicted) # Computes the accuracy.
    return accuracy, precision, recall, f1_metric # Returns the calculated metrics.

def specificity_score(y_true, y_pred): # Defines a function for calculating the specificity score.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # Extracts the true negative (tn), false positive (fp), false negative (fn), and true positive (tp) counts from the confusion matrix.
    return tn / (tn + fp) # Calculates and returns the specificity, which is the proportion of true negatives identified among all negative cases.
    
def plot_loss(epochs, train_losses, val_losses, metric_name, x_label, filename): # Defines a function for plotting the training and validation losses over epochs.
    plt.figure(figsize=(6, 6)) # Creates a new figure for plotting with a specified size.
    plt.plot(epochs, train_losses, 'r', label='Training Loss') # Plots the training losses.
    plt.plot(epochs, val_losses,   'b', label='Validation Loss') # Plots the validation losses.
    plt.title(f'{x_label} vs {metric_name}') # Sets the title, x-label, and y-label of the plot.
    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.grid(True) # Adds a grid to the plot.
    plt.tight_layout() # Adjusts the plot layout.
    plt.legend() # Adds a legend to the plot.
    plt.savefig(filename) # Saves the plot to a file.
    plt.show() # Displays the plot.

def plot_all_metrics(epochs, accuracy_list, f1_list, specificity_list, precision_list, recall_list, title, filename): # Begins the definition of a function for plotting various metrics over epochs.
    plt.figure(figsize=(6, 6)) # Creates a new figure for plotting.
    plt.plot(epochs, accuracy_list,    label='Accuracy', linestyle='-') # calls are made to plot each metric with different line styles for distinction.
    plt.plot(epochs, f1_list,          label='F1-Score', linestyle='--')
    plt.plot(epochs, specificity_list, label='Specificity', linestyle='-.')
    plt.plot(epochs, precision_list,   label='Precision', linestyle=':')
    plt.plot(epochs, recall_list,      label='Recall', linestyle='-.')
    plt.xlabel('Epochs') # Sets the x and y-axis labels.
    plt.ylabel('Metrics')
    plt.title(title) # Sets the title of the plot.
    plt.legend(loc="lower right")
    plt.tight_layout() # Adjusts the plot layout.
    plt.grid(True) # Adds a grid to the plot for better readability.
    plt.savefig(filename) # Saves the plot to the specified file.
    plt.show() # Displays the plot.

def plot_metrics_vs_f1(metric_values, f1_values, metric_name, title, filename): # Defines a function for plotting the relationship between a specific metric and the F1-Score. These functions are vital for visualizing the relationship between different metrics during the training and validation process, offering insights into the model's performance and helping in fine-tuning and evaluation.
    plt.figure() # Creates a new figure for plotting.
    plt.plot(metric_values, f1_values, '-o', label=f'{metric_name} vs F1-Score') # Plots the metric values against the F1-Score.
    plt.title(title) # Set the title, x-axis label, and y-axis label of the plot.
    plt.xlabel(metric_name)
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, class_names): # Defines a function for plotting a confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 6))  # Creates a subplot with a specified size for the confusion matrix.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names) # Uses Seaborn to create a heatmap for the confusion matrix. It annotates each cell with the count and uses a blue color map.
    plt.ylabel('Actual') # Sets the y-axis and x-axis labels.
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('InceptionV3_Base_Confusion_Matrix.png') # Saves the plot to a file.
    plt.show()

def setup_logging(): # Defines a function for setting up logging.
    logging.basicConfig(filename='training_log.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s') # Configures the logging system, specifying the log file, logging level, and message format.

if __name__ == "__main__": # Checks if the script is run as the main program.
    args = get_input() # Calls the get_input function to parse command-line arguments.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Generates a timestamp for the current run, which can be used for logging or saving files with a unique identifier.
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determines whether to use a CUDA-enabled GPU or fall back to CPU based on availability.
    
    if device.type == "cuda": # Checks if the device is a GPU.
        print(f"Using CUDA-enabled GPU: {torch.cuda.get_device_name(0)}") # Prints the name of the GPU being used.
    else: # Fallback in case a GPU is not available.
        print("Using CPU") # Indicates that the CPU will be used for training.
        
    learning_rate        = args.learning_rate
    momentum             = args.momentum
    num_epochs           = args.num_epochs # Sets the number of epochs.
    patience             = args.patience   # Sets the patience for early stopping.
    min_improvement      = args.min_improvement # Sets the minimum improvement required for resetting the patience counter.
    image_resolution     = tuple(map(int, args.image_resolution.split(','))) # Parses the image resolution from a string to a tuple of integers.
    data_dir             = args.data_dir # Sets the directory where the data is located.
    batch_size           = args.batch_size # Sets the batch size for training.
    num_classes          = args.num_classes # Sets the number of classes in the classification task.
    split_ratio          = args.split_ratio # Sets the ratio for splitting the dataset into training and validation sets.
    loss_function_choice = args.loss_function # Retrieves the choice of loss function from the command-line arguments.
    
    # Defines a series of transformations to be applied to the images.
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_resolution),  # Randomly resizes and crops the images to the specified resolution, such as 299x299.
        transforms.RandomRotation(degrees=45), # Randomly rotates the images by up to 45 degrees.
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), #Adjusts the brightness, contrast, and saturation of the images.
        transforms.RandomHorizontalFlip(), # Randomly flips the images horizontally.
        transforms.RandomVerticalFlip(), # Randomly flips the images vertically.
        transforms.ToTensor(), # Converts the images to PyTorch tensors.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizes the tensors using the specified mean and standard deviation values.
    ])
    
    all_dataset = CustomDataset(root_dir=data_dir, transform=transform) # Creates an instance of the CustomDataset class for the entire dataset.
    train_size = int(split_ratio * len(all_dataset)) # Calculates the size of the training dataset based on the split ratio.
    val_size = len(all_dataset) - train_size # Calculates the size of the validation dataset.
    
    if val_size == 0: # Checks if the validation dataset size is zero.
        sys.exit("Validation dataset size is zero. Please check your data split ratio.") # Exits the script if there are no data in the validation set.
    
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size]) # Randomly splits the dataset into training and validation sets.
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Creates a DataLoader for the training dataset, which batches the data and shuffles it.
    valloader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Creates a DataLoader for the validation dataset without shuffling.
    
    model = torchvision.models.inception_v3(pretrained=False, aux_logits=True) # Initializes the Inception V3 model without pre-trained weights and with auxiliary logits.
    
    num_features = model.fc.in_features # Retrieves the number of input features for the final fully connected (fc) layer.
    model.fc = nn.Linear(num_features, num_classes) # Replaces the final fc layer with a new one adjusted for the number of classes.
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes) # Adjusts the auxiliary classifier's fc layer for the number of classes.
    model = model.to(device) # Moves the model to the chosen computing device (GPU or CPU).
    criterion = initialize_loss_function(args.loss_function) # Initializes the loss function based on the user's choice.
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum) # Initializes the SGD optimizer with the learning rate and momentum.
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # Sets up a learning rate scheduler that decreases the learning rate after certain epochs.

    accuracy_list, precision_list, recall_list, f1_list, specificity_list, train_loss_list, val_loss_list = [], [], [], [], [], [], []
        
    epochs_no_improve = 0 # Counter for tracking epochs without improvement.
    best_model_weights = copy.deepcopy(model.state_dict()) # Stores a copy of the best model weights.
    best_loss = np.inf # Initializes the best loss as infinity for comparison.
    
    log_filename = f"InceptionV3_Base_log_{run_timestamp}.txt" # Sets up a filename for logging the training process with a timestamp.

    with open(log_filename, 'w') as log_file: # Opens a new file for logging the training process.
        log_file.write("Epoch\tTraining Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tF1-Score\tSpecificity\n") # Writes the header line for the log file, indicating the metrics that will be logged for each epoch.

    for epoch in range(num_epochs): # Starts a loop over the specified number of training epochs.
    
        try: # Begins a try block to handle any exceptions during training.
            print(f"Starting Epoch {epoch + 1}/{num_epochs}...") # Prints a message indicating the start of an epoch.
            
            running_train_loss = 0.0 # Initializes variables for accumulating training and validation losses and metrics:
            running_val_loss   = 0.0 # Initializes a variable for the running validation loss.
            # Initialize variables for various validation metrics.
            val_acc            = 0.0
            val_precision      = 0.0
            val_recall         = 0.0
            val_f1             = 0.0
            val_specificity    = 0.0
            
            # Training Phase
            model.train() # Puts the model in training mode (this affects layers like dropout, batchnorm, etc.).
            num_train_batches = len(trainloader) # Calculates the number of training batches.
            
            for inputs, labels in trainloader: # Iterates over each batch of training data and labels.
                inputs, labels = inputs.to(device), labels.to(device) # Moves the inputs and labels to the computing device (GPU or CPU).
                optimizer.zero_grad() # Clears the gradients of all optimized tensors.
                outputs = model(inputs) # Passes the inputs through the model and gets the outputs.
                loss_main = criterion(outputs[0], labels) # Computes the main loss between the outputs and the labels.
                loss_aux = criterion(outputs[1], labels) # Computes the auxiliary loss (Inception V3 has an auxiliary output).
                loss = loss_main + 0.4 * loss_aux # Combines the main and auxiliary losses, with the auxiliary loss having a lower weight.
                    
                loss.backward()  # Performs backpropagation to compute gradients.
                optimizer.step() # Updates the model parameters.
                running_train_loss += loss.item() # Accumulates the training loss.

            avg_train_loss = running_train_loss / num_train_batches # Calculates the average training loss for the epoch.

            # Validation Phase
            model.eval() # Puts the model in evaluation mode (affects layers like dropout, batchnorm, etc.).
            num_val_batches = len(valloader) # Calculates the number of validation batches.
            all_labels, all_predictions = [], [] # Initializes lists to store all labels and predictions for the validation set.

            with torch.no_grad(): # Temporarily sets all the requires_grad flags to false, indicating that gradients should not be calculated during the validation phase.
                for inputs, labels in valloader: # Iterates over each batch of validation data and labels.
                    inputs, labels = inputs.to(device), labels.to(device) # Moves the inputs and labels to the computing device.
                    outputs = model(inputs) # Passes the inputs through the model to get outputs.
                    loss = criterion(outputs, labels) # Computes the loss between the outputs and the labels.

                    running_val_loss += loss.item() # Accumulates the validation loss.

                    acc, prec, rec, f1_val = calculate_metrics(outputs, labels) # Calculates various metrics for the validation data.
                    # Accumulates the calculated metrics.
                    val_acc += acc
                    val_precision += prec
                    val_recall += rec
                    val_f1 += f1_val
                    
                    _, predicted = torch.max(outputs, 1) # Determines the class predictions.
                    # Appends the labels and predictions to the respective lists.
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            avg_val_loss = running_val_loss / num_val_batches # Calculates the average validation loss for the epoch.
            val_acc /= num_val_batches # Calculates the average validation accuracy for the epoch.
            # Calculates the average precision, recall, and F1-score for the validation set.
            val_precision /= num_val_batches
            val_recall /= num_val_batches
            val_f1 /= num_val_batches
            val_specificity = specificity_score(all_labels, all_predictions)

            # Step the learning rate scheduler
            scheduler.step() # Adjusts the learning rate according to the scheduler.

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}") # Prints the average training and validation loss for the epoch.
            print(f"Accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1-Score: {val_f1}, Specificity: {val_specificity}") # Prints the calculated accuracy, precision, recall, F1-score, and specificity for the validation set.
            
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
            accuracy_list.append(val_acc)
            precision_list.append(val_precision)
            recall_list.append(val_recall)
            f1_list.append(val_f1)
            specificity_list.append(val_specificity)

            # Log the metrics to the log file
            with open(log_filename, 'a') as log_file: # Opens the log file in append mode.
                log_file.write(f"{epoch + 1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc:.4f}\t{val_precision:.4f}\t{val_recall:.4f}\t{val_f1:.4f}\t{val_specificity:.4f}\n") # Writes the training and validation metrics for the current epoch to the log file.

            # Early stopping logic
            if avg_val_loss < best_loss - min_improvement: # Checks if the average validation loss has improved significantly (defined by min_improvement).
                best_loss = avg_val_loss # Updates the best loss to the current average validation loss.
                best_model_weights = copy.deepcopy(model.state_dict()) # Updates the best model weights.
                epochs_no_improve = 0 # Resets the no-improvement epoch counter.
            else: # If there is no significant improvement.
                epochs_no_improve += 1 # Increments the no-improvement epoch counter.

            if epochs_no_improve == patience: # Checks if the no-improvement counter has reached the patience limit.
                print(f"Early stopping after {epoch + 1} epochs with no improvement.") # Prints a message about early stopping and writes it to the log file.
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"Early stopping after {epoch + 1} epochs with no improvement.\n")
                break # Breaks out of the training loop, effectively stopping the training.

        except Exception as e: # Catches any exceptions that occur during the epoch.
            error_msg = f"Error during Epoch {epoch + 1}: {str(e)}" # Formats an error message.
            print(error_msg) # Prints the error message.
            
            with open(log_filename, 'a') as log_file:
                log_file.write(error_msg + "\n")
            break
 
    model.load_state_dict(best_model_weights) # Loads the best model weights saved during training.
    best_model_name = f"trained_InceptionV3_Base_bestmodel_{run_timestamp}.pth" # Formats the filename for saving the best model.
    torch.save(model.state_dict(), best_model_name) # Saves the best model to a file.
    
    model.eval() # Puts the model in evaluation mode.
    all_labels, all_predictions = [], [] # Initializes empty lists for labels and predictions.
    with torch.no_grad(): # Ensures that no gradients are computed during this phase.
        for inputs, labels in valloader: # Iterates over the validation dataset.
            inputs, labels = inputs.to(device), labels.to(device) # Moves inputs and labels to the computing device.
            outputs = model(inputs) # Passes the inputs through the model to get outputs.

            _, predicted = torch.max(outputs, 1) # Determines the predicted class for each input.
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy()) # Extends the lists with the actual labels and the predicted labels.
    
    class_names = list(all_dataset.class_to_label.keys()) # Retrieves the class names from the dataset.

    classification_rep = classification_report(all_labels, all_predictions, target_names=class_names) # Generates a classification report.
    print(classification_rep) # Prints the classification report.
    cm = confusion_matrix(all_labels, all_predictions) # Computes the confusion matrix.
    plot_confusion_matrix(cm, class_names) # Plots the confusion matrix using the previously defined function.

    epochs = range(1, len(train_loss_list) + 1) # Creates a range of epoch numbers.
    plot_loss(epochs, train_loss_list, val_loss_list, 'Loss', 'Epoch', f'InceptionV3_Base_training_validation_loss_{run_timestamp}.png') # Calls the plot_loss function to plot the training and validation losses over epochs. The filename for saving the plot includes a timestamp for uniqueness.

    plot_all_metrics(epochs, accuracy_list, f1_list, specificity_list, precision_list, recall_list, "Performance Over Epochs", f'InceptionV3_Base_all_metrics_{run_timestamp}.png') #Calls the plot_all_metrics function to plot various performance metrics (accuracy, F1-score, specificity, precision, and recall) over epochs. The filename for the plot includes a timestamp.
        
    metrics_for_comparison = [accuracy_list, recall_list, precision_list, specificity_list] #Prepares a list of metrics for comparison.
    metric_names_for_comparison = ['Accuracy', 'Recall', 'Precision', 'Specificity'] #Prepares the corresponding names of the metrics.

    for metric, metric_name in zip(metrics_for_comparison, metric_names_for_comparison): #Loops over each metric and its name.
        plot_metrics_vs_f1(metric, f1_list, metric_name,
                           f'{metric_name} vs F1-Score', f'InceptionV3_Base_{metric_name}_Vs_F1-Score_{run_timestamp}.png') #Calls the plot_metrics_vs_f1 function to plot each metric against the F1-score. The filename for each plot includes the metric name and a timestamp.
        

# Improving the classification accuracy and overall performance of a convolutional neural network like Inception V3 involves several strategies.
# Here are detailed and comprehensive suggestions:

#1. Data Augmentation:
#a) Expand the dataset with more varied images, if possible.
#b) Increase the diversity of transformations in data preprocessing (e.g., more aggressive rotations, flips, scaling, or even adding noise).
#c) Use advanced augmentation techniques like Mixup, CutMix, or GAN-generated images.

#2. Hyperparameter Tuning:
#a) Experiment with different learning rates and schedules. Consider using learning rate finders to identify the optimal starting learning rate.
#b) Adjust the batch size based on the available memory. Try different optimizers (e.g., Adam, RMSprop) and their parameters (like different momentums or decay rates).
#c) Tune other hyperparameters such as the weight for the auxiliary loss in Inception V3.

#3. Model Architecture Tweaks:
#a) Adjust the number of neurons or layers in the fully connected part of the network.
#b) Experiment with different activation functions (e.g., LeakyReLU, ELU).
#c) Try using dropout or batch normalization in the fully connected layers to prevent overfitting.

#4. Regularization Techniques:
#a) Implement dropout layers to prevent overfitting.
#b) Use L1/L2 regularization in the loss function.
#c) Experiment with different early stopping criteria.

#5. Loss Function Optimization:
#a) If dealing with imbalanced classes, use a weighted loss function.
#b) Experiment with different loss functions (e.g., focal loss for imbalanced datasets).

#6. Ensemble Methods:
#a) Train multiple models and use their averaged predictions. Use a voting system among different models.

#7. Advanced Training Techniques:
#a) Implement gradient accumulation if limited by smaller batch sizes.
#b) Use techniques like Knowledge Distillation for training more compact models.

#8. Experiment with Transfer Learning:
#Start with a model pre-trained on a large dataset (like ImageNet) and fine-tune it on your specific dataset.

#9. Data Quality and Preprocessing:
#a) Ensure the data is clean and well-labeled.
#b) Normalize or standardize the data appropriately.
#c) Experiment with different image sizes and aspect ratios.

#10. Evaluation Metrics:
#Apart from accuracy, focus on other metrics like precision, recall, F1-score, especially if the classes are imbalanced.
#Use ROC-AUC curves and Precision-Recall curves for a more comprehensive understanding of model performance.

#11. Feature Engineering:
#Explore additional features that could be extracted from the images and used alongside the raw pixel data.
#Use techniques like PCA (Principal Component Analysis) for dimensionality reduction in cases of high-dimensional data.

#12. Post-processing Techniques:
#Implement custom rules based on domain knowledge to refine predictions.
#Use model explainability tools (like SHAP or LIME) to understand model predictions and refine the model accordingly.

#13. Training Infrastructure:
#Utilize mixed precision training for faster computation.
#If training on multiple GPUs, ensure efficient data parallelism.
#These suggestions cover a wide range of aspects from data handling to model architecture and training strategies. It's important to iteratively experiment and validate the impact of these changes, as the effectiveness of each technique can vary depending on the specific context and data.
