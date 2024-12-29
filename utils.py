import torch
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_experts():
    expert1 = models.densenet121(pretrained=True).to(device)
    expert1.classifier = nn.Linear(expert1.classifier.in_features, 4).to(device)
    expert2 = models.densenet161(pretrained=True).to(device)
    expert2.classifier = nn.Linear(expert2.classifier.in_features, 4).to(device)
    expert3 = models.densenet201(pretrained=True).to(device)
    expert3.classifier = nn.Linear(expert3.classifier.in_features, 4).to(device)
    
    expert1.load_state_dict(torch.load('./checkpoints/expert1.pth')['model'])
    expert2.load_state_dict(torch.load('./checkpoints/expert2.pth')['model'])
    expert3.load_state_dict(torch.load('./checkpoints/expert3.pth')['model'])
    
    return [expert1, expert2, expert3]