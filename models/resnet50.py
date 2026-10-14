import timm
import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = timm.create_model('resnet50', pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, config["num_classes"])
        self.dropout = nn.Dropout(0.3)

    def configure_trainable_params(self):
        for name, param in self.named_parameters():
            if "model" in name:
                param.requires_grad = False

    @property
    def trainable_parameters(self):
        total = 0
        for param in self.parameters():
            if param.requires_grad:
                total += param.numel()
        return total
    
    def forward_features(self, img: torch.Tensor):
        if len(img.shape) == 4:
            feats = self.model(img)
            
        elif len(img.shape) == 5:
            raise NotImplementedError

        return feats        

    def forward(self, img: torch.Tensor):
        feats = self.model(img)
        feats = self.dropout(feats)

        logits = self.fc(feats)

        return logits
    

