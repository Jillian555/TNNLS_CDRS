import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
from torch.nn.parameter import Parameter


class GMM(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, cluster_num):
        super(GMM, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc_semi = GraphConvolution(hidden_dim1, cluster_num, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.cluster_num = cluster_num
        self.pi_prior = Parameter(torch.ones(cluster_num) / cluster_num)
        self.mean_prior = Parameter(torch.zeros(cluster_num, hidden_dim2))
        self.log_var_prior = Parameter(torch.randn(cluster_num, hidden_dim2))

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def semi(self, x, adj):
        hidden1 = self.gc1(x, adj)
        output = self.gc_semi(hidden1, adj)
        return F.log_softmax(output, dim=1), F.softmax(output, dim=1), output

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def predict(self, z):
        estim_pro = torch.cat([-0.5 * torch.sum(np.log(np.pi * 2) + self.log_var_prior[i].unsqueeze(0) \
                                                + (z - self.mean_prior[i].unsqueeze(0)).pow(2) / \
                                                torch.exp(self.log_var_prior[i].unsqueeze(0)), 1).unsqueeze(1) \
                               for i in range(self.cluster_num)], 1)
        cluster_pro_back = torch.exp(torch.log(self.pi_prior.unsqueeze(0)) + estim_pro)
        rst = torch.argmax(cluster_pro_back, dim=1).cpu().numpy()
        rst_score = torch.max(cluster_pro_back, dim=1)[0]
        sumed = torch.sum(cluster_pro_back, dim=1)
        cluster_pro_back2 = torch.zeros_like(cluster_pro_back)
        for i in range(cluster_pro_back.shape[0]):
            rst_score[i] /= sumed[i]
            for j in range(cluster_pro_back.shape[1]):
                cluster_pro_back2[i][j] = cluster_pro_back[i][j] / sumed[i]
        return rst, rst_score, cluster_pro_back, cluster_pro_back2

    def elbo_loss(self, mean, std):
        pi = self.pi_prior
        mean_c = self.mean_prior
        log_std_pro = self.log_var_prior
        det = 1e-10
        normlize = 0.01
        z = self.reparameterize(mean, std)
        estim_prior_pro = torch.cat([-0.5 * torch.sum(np.log(np.pi * 2) + self.log_var_prior[i].unsqueeze(0) \
                                                      + (z - self.mean_prior[i].unsqueeze(0)).pow(2) / \
                                                      torch.exp(self.log_var_prior[i].unsqueeze(0)), 1).unsqueeze(1) \
                                     for i in range(self.cluster_num)], 1)

        cluster_pro_prior = torch.exp(torch.log(self.pi_prior.unsqueeze(0)) + estim_prior_pro) + det
        cluster_pro_prior = cluster_pro_prior / (cluster_pro_prior.sum(1).unsqueeze(1))
        Loss = 0.5 * torch.mean(
            torch.sum(cluster_pro_prior * torch.sum(self.log_var_prior.unsqueeze(0) + torch.exp(std.unsqueeze(1) \
                                                                                                - self.log_var_prior.unsqueeze(
                0)) + (mean.unsqueeze(1) - self.mean_prior.unsqueeze(0)).pow(2) \
                                                    / torch.exp(self.log_var_prior.unsqueeze(0)), 2), 1))

        Loss -= torch.mean(torch.sum(cluster_pro_prior * torch.log(self.pi_prior.unsqueeze(0) / (cluster_pro_prior)),
                                     1)) + 0.5 * torch.mean(torch.sum(1 + std, 1))
        Loss *= normlize
        return Loss

    def forward(self, x, adj):
        mean, logvar = self.encode(x, adj)
        z = self.reparameterize(mean, logvar)
        logsm_semi, sm_semi, _semi = self.semi(x, adj)
        return self.dc(z), mean, logvar, z, logsm_semi, sm_semi, _semi


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
