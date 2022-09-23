# -*-coding:utf-8-*-
import argparse
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

# load module
from model import GMM
from optimizer import loss_function
from utils import load_data, mask_gae_edges, preprocess_graph, load_ds
from metric import cluster_accuracy

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gmm-vgae', help="Model used.")
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--func', type=str, default='None', help='The policy for selecting pseudo-labels:[col,s,col-s,'
                                                             'col-e,col-m,dw,all].')
parser.add_argument('--e', type=int, default=100, help='Number of pretrained epochs.')
parser.add_argument('--e1', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e2', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e3', type=int, default=300, help='Number of epochs.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--update', type=int, default=10, help='update epoch.')
parser.add_argument('--pseudo_num', type=int, default=25, help='The number of pseudo-labels for each class.')
parser.add_argument('--max_num', type=int, default=300, help='Maximum number of nodes.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.011, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--w_sup', type=float, default=0.5, help='Weight of loss_sup.')
parser.add_argument('--ds', type=str, default='cora', help='Type of dataset.')
parser.add_argument('--sds', type=int, default=0, help='whether to use small dataset, 0/1.')
args = parser.parse_args()


def main():
    SEED = 20
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # load data
    if args.sds == 1:
        adj, features, labels = load_ds(args.ds)
    else:
        adj, features, _, _, labels = load_data(args.ds)
    if args.func == 'dw':
        alpha = 1e-6
        A_tilde = adj.toarray() + np.identity(adj.shape[0])
        D = A_tilde.sum(axis=1)
        A_ = np.array(np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5)))
        Lambda = np.identity(len(A_))
        L = np.diag(D) - adj
        from numpy.linalg import inv
        P = inv(L + alpha * Lambda)

    cluster_num = labels.max().item() + 1
    n_nodes, feat_dim = features.shape
    adj_train = mask_gae_edges(adj)
    adj_norm = preprocess_graph(adj_train)
    adj_label = adj + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # build model
    gmm = GaussianMixture(n_components=cluster_num, covariance_type='diag', random_state=0)
    model = GMM(feat_dim, args.hidden1, args.hidden2, args.dropout, cluster_num)

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pseudo_labels = torch.full(labels.shape, -1, dtype=torch.long)
    idx_train = set()
    loss_sup = 0.
    accs = []
    saccs = []
    pnum = []

    if args.gpu != -1:
        features = features.cuda()
        adj_norm = adj_norm.cuda()
        adj_label = adj_label.cuda()
        pos_weight = pos_weight.cuda()
        labels = labels.cuda()
        model = model.cuda()
        pseudo_labels = pseudo_labels.cuda()

    for epoch in range(int(args.epochs) + 1):
        print(epoch)
        model.train()
        if epoch < args.e:
            optimizer.zero_grad()
            recovered, mean, logvar, z, semi_out, _, _ = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            hidden_emb = mean.cpu().data.numpy()
            cluster_pred = gmm.fit_predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Pretrained model result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            if args.gpu != -1:
                model.pi_prior.data = torch.from_numpy(gmm.weights_).cuda().float()
                model.mean_prior.data = torch.from_numpy(gmm.means_).cuda().float()
                model.log_var_prior.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
            else:
                model.pi_prior.data = torch.from_numpy(gmm.weights_).float()
                model.mean_prior.data = torch.from_numpy(gmm.means_).float()
                model.log_var_prior.data = torch.log(torch.from_numpy(gmm.covariances_).float())

        else:
            optimizer.zero_grad()
            recovered, mean, logvar, z, semi_out, _, _ = model(features, adj_norm)
            cluster_pred, cluster_pred_score, score, score2 = model.predict(z)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Trained model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
            accs.append(float('{:.4f}'.format(acc)))

            loss_re = loss_function(preds=recovered, labels=adj_label, n_nodes=n_nodes, norm=norm,
                                    pos_weight=pos_weight)
            if epoch == args.e1:
                max_p_indc = torch.argsort(cluster_pred_score, descending=True)
                cls_sort = {}
                for i in max_p_indc:
                    if cluster_pred[i].item() not in cls_sort:
                        cls_sort[cluster_pred[i].item()] = [i.item()]
                    else:
                        cls_sort[cluster_pred[i].item()].append(i.item())
                for key in cls_sort.keys():
                    j = 0
                    for i in cls_sort[key]:
                        if j < args.pseudo_num:
                            if i not in idx_train:
                                idx_train.add(i)
                                pseudo_labels[i] = key
                                j += 1
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()

            if epoch > args.e2:
                if args.func == 'all':
                    loss_sup = F.nll_loss(semi_out, cluster_pred.detach())
                else:
                    loss_sup = F.nll_loss(semi_out[idx_train], pseudo_labels[idx_train])

            cluster_pred = torch.tensor(cluster_pred)
            if args.gpu != -1:
                cluster_pred = cluster_pred.cuda()
            if args.func == 'col' and epoch > args.e3 and epoch % args.update == 0:
                num = 0
                _, nc_pred = torch.max(torch.exp(semi_out), 1)
                idx_train = set(idx_train.numpy())
                for i in range(n_nodes):
                    if cluster_pred[i] == nc_pred[i]:
                        if i not in idx_train and num != args.max_num:
                            idx_train.add(i)
                            num += 1
                            pseudo_labels[i] = cluster_pred[i]
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()
            if args.func == 's' and epoch > args.e3 and epoch % args.update == 0:
                max_p_indc = torch.argsort(cluster_pred_score, descending=True)
                num = 0
                idx_train = set(idx_train.numpy())
                for i in max_p_indc:
                    i = i.item()
                    if i not in idx_train and num != args.max_num:
                        idx_train.add(i)
                        num += 1
                        pseudo_labels[i] = cluster_pred[i]
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()
            if args.func == 'col-s' and epoch > args.e3 and epoch % args.update == 0:
                max_p_indc = torch.argsort(cluster_pred_score, descending=True)
                num = 0
                _, nc_pred = torch.max(torch.exp(semi_out), 1)
                idx_train = set(idx_train.numpy())
                for i in max_p_indc:
                    i = i.item()
                    if cluster_pred[i] == nc_pred[i]:
                        if i not in idx_train and num != args.max_num:
                            idx_train.add(i)
                            num += 1
                            pseudo_labels[i] = cluster_pred[i]
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()
            if args.func == 'col-e' and epoch > args.e3 and epoch % args.update == 0:
                entropy_score = torch.zeros(n_nodes)
                for i in range(n_nodes):
                    for x in score2[i]:
                        entropy_score[i] += (-x) * torch.log(x)
                max_p_indc = torch.argsort(entropy_score)
                num = 0
                _, nc_pred = torch.max(torch.exp(semi_out), 1)
                idx_train = set(idx_train.numpy())
                for i in max_p_indc:
                    i = i.item()
                    if cluster_pred[i] == nc_pred[i]:
                        if i not in idx_train and num != args.max_num:
                            idx_train.add(i)
                            num += 1
                            pseudo_labels[i] = cluster_pred[i]
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()
            if args.func == 'col-m' and epoch > args.e3 and epoch % args.update == 0:
                margin_score = torch.zeros(n_nodes)
                margins, indices = score2.topk(2, dim=1, largest=True, sorted=True)
                for i in range(n_nodes):
                    margin_score[i] = margins[i][0] - margins[i][1]
                max_p_indc = torch.argsort(margin_score, descending=True)
                num = 0
                _, nc_pred = torch.max(torch.exp(semi_out), 1)
                idx_train = set(idx_train.numpy())
                for i in max_p_indc:
                    i = i.item()
                    if cluster_pred[i] == nc_pred[i]:
                        if i not in idx_train and num != args.max_num:
                            idx_train.add(i)
                            num += 1
                            pseudo_labels[i] = cluster_pred[i]
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()
            if args.func == 'dw' and epoch > args.e3 and epoch % args.update == 0:
                idx_train = set(idx_train.numpy())
                for k in range(cluster_num):
                    nodes = (pseudo_labels == k).nonzero().squeeze().cpu()
                    probability = P[:, nodes].sum(axis=1).flatten()
                    for i in np.argsort(probability).tolist()[0][::-1][:50]:
                        if i in idx_train:
                            continue
                        idx_train.add(i)
                        pseudo_labels[i] = k
                idx_train = list(idx_train)
                idx_train = torch.tensor(idx_train, dtype=torch.int32).long()

            if epoch % args.update == 0 and epoch > args.e2:
                pnum.append(len(list(idx_train)))
                sacc, _, _ = cluster_accuracy(pseudo_labels[idx_train].cpu(), labels[idx_train].cpu(), cluster_num)
                saccs.append(float('{:.4f}'.format(sacc)))

            elbo = model.elbo_loss(mean, logvar)
            loss = loss_re + elbo + args.w_sup * loss_sup
            print('loss: {:.4f} {:.4f} {:.4f}'.format(loss_re, elbo, loss_sup))
            loss.backward()
            optimizer.step()

        model.eval()
        recovered, mean, logvar, z, semi_out, _, _ = model(features, adj_norm)
        cluster_pred, _, _, _ = model.predict(z)
        acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
        print('Eval model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

    print(args)
    print('Cluster accuracy:', accs)
    print('Pseudo-label accuracy:', saccs)
    print('Number of pseudo-labels:', pnum)


def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    init()
    main()
