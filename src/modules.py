import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class NonLinear(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0., gain=1.):
        super(NonLinear, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_mask=None):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
        
class ResNetSimple(nn.Module):
    def __init__(self, device):
        super(ResNetSimple, self).__init__()
        self.base = torchvision.models.video.r2plus1d_18(pretrained=True).to(device)
#         self.proj = nn.Linear(400, 2)
        self.proj = NonLinear(400, 128, 2, dropout=0.1)
        dfs_freeze(self.base)
        
    def forward(self, uid, video, label):
        out = self.base(video)
        out = self.proj(out)
        return out