import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import *
from moe import MoE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

criterion = nn.CrossEntropyLoss()

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

if __name__ == '__main__':
    experts = load_experts()
    moe_model = MoE(experts).to(device)
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=0.001)
    train_moe(moe_model, criterion, optimizer, train_loader, val_loader, 'moe', num_epochs=10)