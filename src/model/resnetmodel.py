import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ResnetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.resnet = resnet18(pretrained=True)
        prev_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(prev_in_features, self.n_classes)

        y = 1 / np.sqrt(self.resnet.fc.in_features)
        self.resnet.fc.weight.data.normal_(0, y)

    def forward(self, x):
        return self.resnet(x)

    def block_backbone_update(self):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for p in self.resnet.fc.parameters():
            p.requires_grad = True

    def unblock_backbone_update(self):
        for p in self.resnet.parameters():
            p.requires_grad = True

    def compute_all(self, batch):
        x = batch['image']
        y = batch['label']
        logits = self.resnet(x)

        loss = F.cross_entropy(logits, y)
        
        prediction = logits.argmax(axis=1).cpu().numpy().reshape(-1)
        target = y.cpu().numpy().reshape(-1)

        acc = accuracy_score(target, prediction)
        f1_micro = f1_score(target, prediction, average="micro")
        f1_macro = f1_score(target, prediction, average="macro")

        metrics = dict(acc=acc, f1_micro=f1_micro, f1_macro=f1_macro)

        return loss, metrics
