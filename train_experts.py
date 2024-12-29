import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
from moe import *

train_dir = './datasets/brain_tumor_detection/figshare/Training'
test_dir = './datasets/brain_tumor_detection/figshare/Testing'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=test_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, train_loader, val_loader, save_path, num_epochs=10):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader):
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        val_accuracy = evaluate_model(model, val_loader, log=False)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    torch.save({'model': model.state_dict()}, f'./checkpoints/{save_path}.pth')

    return train_losses, val_accuracies

def train_moe(model, criterion, optimizer, train_loader, val_loader, save_path, num_epochs=10):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader):
            labels = labels.to(device)
            images = images.to(device) 
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        val_accuracy = evaluate_model(model, val_loader, log=False)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    torch.save({'model': model.state_dict()}, f'./checkpoints/{save_path}.pth')

    return train_losses, val_accuracies

def evaluate_model(model, val_loader, log=True):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total

    if log:
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy

expert1 = models.densenet121(pretrained=True).to(device)
expert1.classifier = nn.Linear(expert1.classifier.in_features, 4).to(device)
expert2 = models.densenet161(pretrained=True).to(device)
expert2.classifier = nn.Linear(expert2.classifier.in_features, 4).to(device)
expert3 = models.densenet201(pretrained=True).to(device)
expert3.classifier = nn.Linear(expert3.classifier.in_features, 4).to(device)

optimizer_expert1 = optim.Adam(expert1.parameters(), lr=0.001)
optimizer_expert2 = optim.Adam(expert2.parameters(), lr=0.001)
optimizer_expert3 = optim.Adam(expert3.parameters(), lr=0.001)

print("Training Expert 1")
train_model(expert1, criterion, optimizer_expert1, train_loader, val_loader, save_path='expert1', num_epochs=10)
print("Training Expert 2")
train_model(expert2, criterion, optimizer_expert2, train_loader, val_loader, save_path='expert2', num_epochs=10)
print("Training Expert 3")
train_model(expert3, criterion, optimizer_expert3, train_loader, val_loader, save_path='expert3', num_epochs=10)

moe_model = MoE([expert1, expert2, expert3]).to(device)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001)
print("Training MoE")
train_moe(moe_model, criterion, optimizer_moe, train_loader, val_loader, save_path='moe', num_epochs=1)