import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前训练设备: {device}")

class DqnNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(inputs, 512)
        self.l2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.l1(x))
        x = self.l2(x)
        return x
