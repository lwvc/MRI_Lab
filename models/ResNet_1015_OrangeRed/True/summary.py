import torch
from torchinfo import summary
from model import BrainAgeResNet

model = BrainAgeResNet(metadata="both", dropout=0.2)
model.eval()

summary(
    model,
    input_data=[
        torch.zeros(1, 1, 193, 229, 193),
        torch.zeros(1, 1),
        torch.zeros(1, 1)
    ],
    depth=4,
    device="cpu"
)