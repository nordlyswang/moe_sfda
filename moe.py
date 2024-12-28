import torch
from torch import nn

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
        return torch.softmax(self.layer4(x), dim=1)

# 定义MoE模型
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
        return torch.sum(outputs * weights, dim=2)