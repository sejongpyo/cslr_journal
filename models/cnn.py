import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet18, googlenet
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet34(nn.Module):
    def __init__(self, cnn_pretrain):
        super(ResNet34, self).__init__()
        cnn = resnet18(pretrained=cnn_pretrain)
        # cnn = resnet34(pretrained=cnn_pretrain)
        # cnn = googlenet(pretrained=cnn_pretrain)
        self.extractor = nn.Sequential(*list(cnn.children())[:-1])
        # self.extractor = getattr(models, "resnet34")(pretrained=cnn_pretrain)
        # self.extractor.fc = Identity()
        
    def forward(self, x, len_x, lstm=False):
        bsz, seq_len, c, h, w = x.shape
        inputs = x.reshape(bsz*seq_len, c, h, w)
        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.extractor(x).reshape(x.size(0), -1)
        # x = self.extractor(x)
        x = torch.cat([self.pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        
        if lstm:
            x = x.reshape(bsz, seq_len, -1).transpose(0, 1)
        else:
            x = x.reshape(bsz, seq_len, -1).transpose(1, 2)
        
        return x

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])