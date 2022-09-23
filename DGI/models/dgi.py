import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_c, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h1, activation)
        self.gcn_2 = GCN(n_h1, n_h2, activation)
        self.gcn_semi = GCN(n_h1, n_c, lambda x: x)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h2)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj)
        h_11 = self.gcn_2(h_1, adj)
        h_semi = self.gcn_semi(h_1, adj)

        c = self.sigm(self.read(h_11, msk))

        h_2 = self.gcn(seq2, adj)
        h_22 = self.gcn_2(h_2, adj)

        ret = self.disc(c, h_11, h_22, samp_bias1, samp_bias2)
        return ret, h_11, h_semi

    # Detach the return variables
    def embed(self, seq, adj, msk):
        h_1 = self.gcn(seq, adj)
        h_11 = self.gcn_2(h_1, adj)
        c = self.read(h_11, msk)
        return h_11.detach(), c.detach()
