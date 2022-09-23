import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class CDRS(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, cluster_num):
        super(CDRS, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.gc_semi = GraphConvolution(hidden_dim1, cluster_num, dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def semi(self, x, adj):
        hidden1 = self.gc1(x, adj)
        output = self.gc_semi(hidden1, adj)
        return F.log_softmax(output, dim=1), F.softmax(output, dim=1), output

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        logsm_semi, sm_semi, _semi = self.semi(x, adj)
        return self.dc(z), mu, logvar, z, logsm_semi, sm_semi, _semi


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
