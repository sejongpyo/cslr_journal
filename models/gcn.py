import torch
from torch import nn

class BN1d(nn.Module):
    def __init__(self, out_dim, use_bn):
        super(BN1d, self).__init__()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim)
        
    def forward(self, x):
        if not self.use_bn:
            return  x
        origin_shape = x.shape
        x = x.view(-1, origin_shape[-1])
        x = self.bn(x)
        x = x.view(origin_shape)
        return x
    
class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn):
        super(GConv, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        if use_bn:
            self.bn = BN1d(output_dim, use_bn)
        else:
            self.bn = nn.LayerNorm(output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, X, A):
        x = self.fc(X)
        x = torch.matmul(A, x)
        x = self.relu(self.bn(x))
        return x, A
    
if __name__=="__main__":
    sample = torch.randn(2, 10, 512)
    model = GConv(512, 256, True)
    adj = torch.eye(10).unsqueeze(0)
    model(sample, adj)