import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in sorted(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_label = {classname: index for index, classname in enumerate(self.classes)}
        self.image_paths = [os.path.join(root_dir, classname, filename)
                            for classname in self.classes
                            for filename in os.listdir(os.path.join(root_dir, classname))
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path

def load_model(model_path, num_classes):
    model = torchvision.models.inception_v3(pretrained=False, aux_logits=True)  # Set aux_logits to True
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    
    model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(zip(paths, predicted.cpu().numpy()))
    return predictions

def plot_confusion_matrix(cm, class_names, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    test_data_dir = ""  # Specify your test data directory
    model_path = "trained_InceptionV3_Base_bestmodel_20231127_174942.pth"    # Specify your model path
    batch_size = 10
    num_classes = 2  # Update this with your number of classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = InferenceDataset(root_dir=test_data_dir, transform=transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_path, num_classes).to(device)

    predictions = predict(model, testloader, device)

    # Save predictions to a text file
    with open('predictions.txt', 'w') as f:
        for path, pred in predictions:
            f.write(f"{path}: {test_dataset.classes[pred]}\n")

    # Confusion Matrix
    true_labels = [test_dataset.class_to_label[os.path.basename(os.path.dirname(p))] for p, _ in predictions]
    pred_labels = [pred for _, pred in predictions]
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, test_dataset.classes, 'confusion_matrix.png')

    # Classification Report
    print(classification_report(true_labels, pred_labels, target_names=test_dataset.classes))
