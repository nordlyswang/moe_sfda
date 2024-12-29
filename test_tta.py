import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径和转换
test_dir = './datasets/brain_tumor_detection/kaggle/Testing'

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes
print(f"Classes: {class_names}")

# 定义Gating和MoE模型
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)
        # return torch.softmax(self.layer4(x), dim=1)
        weights = torch.softmax(self.layer4(x), dim=1)

        noise = torch.randn_like(weights) * 0.01
        return torch.softmax(weights + noise, dim=1)

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        self.gating = Gating(224*224*3, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        gating_input = x.view(batch_size, -1)
        weights = self.gating(gating_input)
        
        # 获取专家输出
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        weights = weights.unsqueeze(1).expand_as(outputs)
        
        # 聚合专家输出
        return torch.sum(outputs * weights, dim=2), weights

# 加载训练好的专家模型
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

# 加载MoE模型
def load_moe_model():
    experts = load_experts()
    moe_model = MoE(experts).to(device)
    moe_model.load_state_dict(torch.load('./checkpoints/moe.pth')['model'])
    return moe_model

# 计算熵
def calculate_entropy(weights):
    """
    计算门控网络权重的熵
    参数:
        weights: [batch_size, num_experts] 的权重分布
    返回:
        平均熵 (batch 内)
    """
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)  # 避免 log(0)
    return entropy.mean()  # 返回批量样本的平均熵

# 测试时间适应 (TTA) - 最小化熵约束
def test_time_adaptation(model, images, num_adaptation_steps=50, lr=0.5):
    """
    在测试时间引入熵约束，通过调整模型的门控网络权重进行适应。
    """
    model.eval()

    # 确保仅优化门控网络的参数
    for param in model.experts.parameters():
        param.requires_grad = False
    for param in model.gating.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(model.gating.parameters(), lr=lr)

    for step in range(num_adaptation_steps):
        with torch.enable_grad():
            optimizer.zero_grad()
            
            # 前向传播
            _, weights = model(images)
            entropy = calculate_entropy(weights)

            # 调试信息
            print(f"Step {step+1}, Entropy: {entropy.item()}, Weights Mean: {weights.mean().item()}")
            
            # 放大熵以增强梯度更新
            loss = entropy * 1e6

            # 加入正则化约束
            l2_regularization = torch.sum(weights ** 2)
            loss += 0.01 * l2_regularization

            # 优化
            loss.backward()
            optimizer.step()

    # 返回调整后的预测
    with torch.no_grad():
        outputs, _ = model(images)
    return outputs



# 评估MoE模型，带TTA
def evaluate_model_with_tta(model, test_loader, num_adaptation_steps=10, lr=0.5):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    total_entropy = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = test_time_adaptation(model, images, num_adaptation_steps, lr)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            _, weights = model(images)
            total_entropy += calculate_entropy(weights).item()

    accuracy = 100 * correct / total
    avg_entropy = total_entropy / len(test_loader)
    return accuracy, avg_entropy, all_labels, all_preds

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds

def evaluate_experts(experts, test_loader):
    expert_accuracies = {}
    all_labels = []
    all_preds = []
    
    for i, expert in enumerate(experts, start=1):
        print(f"Evaluating Expert {i}...")
        expert_accuracy, expert_labels, expert_preds = evaluate_model(expert, test_loader)
        expert_accuracies[f'Expert {i}'] = expert_accuracy
        
        # 汇总所有专家的结果
        all_labels.extend(expert_labels)
        all_preds.extend(expert_preds)
    
    return expert_accuracies, all_labels, all_preds

# 主测试代码
if __name__ == '__main__':
    # 加载 MoE 模型
    moe_model = load_moe_model()

    # 加载专家模型
    experts = load_experts()

    # 评估专家模型
    print("\nEvaluating Experts...")
    expert_accuracies, _, _ = evaluate_experts(experts, test_loader)
    for expert, accuracy in expert_accuracies.items():
        print(f"{expert} Accuracy: {accuracy:.2f}%")

    # 评估 MoE 模型 (带 TTA 和熵约束)
    print("\nEvaluating MoE Model with TTA...")
    moe_accuracy, moe_avg_entropy, moe_labels, moe_preds = evaluate_model_with_tta(
        moe_model, test_loader, num_adaptation_steps=10, lr=0.5
    )
    print(f"MoE Model Test Accuracy with TTA: {moe_accuracy:.2f}%")
    print(f"MoE Model Average Entropy with TTA: {moe_avg_entropy:.4f}")

    # 打印分类报告和混淆矩阵
    print("MoE Classification Report with TTA:")
    print(classification_report(moe_labels, moe_preds, target_names=class_names))
    cm = confusion_matrix(moe_labels, moe_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
