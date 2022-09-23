import torch.nn as nn
from layers import *
import torch.nn.functional as F


class LinTrans(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, cluster_num):
        super(LinTrans, self).__init__()
        self.fc1 = nn.Linear(input_feat_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_semi = nn.Linear(hidden_dim1, cluster_num)
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x):
        out = F.relu(self.fc1(x))

        out1 = self.fc2(out)
        out1 = self.scale(out1)
        out1 = F.normalize(out1)

        out2 = self.fc_semi(out)
        out2 = self.scale(out2)
        out2 = F.normalize(out2)
        return out1, out2
