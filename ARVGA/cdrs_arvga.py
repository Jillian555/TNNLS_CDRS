from __future__ import division
from __future__ import print_function
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import argparse
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import KMeans
from model import CDRS, Discriminator
from optimizer import loss_function, dis_loss
from utils import load_data, mask_test_edges, preprocess_graph, weights_init, load_ds
from metric import cluster_accuracy
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='cora', help='Type of dataset.')
parser.add_argument('--func', type=str, default='None', help='The policy for selecting pseudo-labels:[col,s,col-s,'
                                                             'col-e,col-m,dw,all].')
parser.add_argument('--pseudo_num', type=int, default=25, help='The number of pseudo-labels for each class.')
parser.add_argument('--max_num', type=int, default=300, help='Maximum number of nodes.')
parser.add_argument('--w_sup', type=float, default=0.5, help='Weight of loss_sup.')
parser.add_argument('--w_re', type=float, default=0.01, help='Weight of loss_re.')
parser.add_argument('--w_pq', type=float, default=1, help='Weight of loss_pq.')
parser.add_argument('--lr_dec', type=float, default=0.11, help='Learning rate.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr_dis', type=float, default=0.001, help='Discriminator learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--update', type=int, default=10, help='update epoch.')
parser.add_argument('--e', type=int, default=100, help='Number of pretrained epochs.')
parser.add_argument('--e1', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e2', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e3', type=int, default=300, help='Number of epochs.')
parser.add_argument('--epoch', type=int, default=400, help='Number of epochs to save models.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--model', type=str, default='arvga', help="Models used")
parser.add_argument('--optimi', type=str, default='SGD', help="Optimizers used:[SGD ADAM]")
parser.add_argument('--arga', type=int, default=1, help='whether to use arga, 0/1.')
parser.add_argument('--save_ckpt', type=int, default=0, help='whether to save checkpoint, 0/1.')
parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use checkpoint, 0/1.')
parser.add_argument('--sds', type=int, default=0, help='whether to use small dataset, 0/1.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
args = parser.parse_args()


def main():
    SEED = 42
    random.seed(SEED)
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
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    pseudo_labels = torch.full(labels.shape, -1, dtype=torch.long)
    idx_train = set()
    loss_sup = 0.0
    cluster_centers = None
    accs = []
    saccs = []
    pnum = []

    # build model
    # This is the Pytorch implementation for ARVGA
    if args.model == 'arvga':
        model = CDRS(feat_dim, args.hidden1, args.hidden2, args.dropout, cluster_num)
    modeldis = Discriminator(args.hidden1, args.hidden2, args.hidden3)
    modeldis.apply(weights_init)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_dis = torch.optim.Adam(modeldis.parameters(), lr=args.lr_dis, betas=(0.5, 0.9))

    if args.gpu != -1:
        pos_weight = pos_weight.cuda()
        features = features.cuda()
        adj_norm = adj_norm.cuda()
        adj_label = adj_label.cuda()
        model = model.cuda()
        modeldis = modeldis.cuda()
        pseudo_labels = pseudo_labels.cuda()

    ee = 0
    if args.use_ckpt == 1:
        ee = args.e
        model.load_state_dict(torch.load(f'./pretrain/{args.model}_{args.ds}_{args.epoch}.pkl'))
        model.eval()
        recovered, mu, logvar, z, semi_out, _, _ = model(features, adj_norm)
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(mu.cpu().data)
        if args.gpu != -1:
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                                       requires_grad=True)
        else:
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)),
                                       requires_grad=True)
        if args.optimi == 'SGD':
            optimizer_dec = torch.optim.SGD(list(model.parameters()) + [cluster_centers], lr=args.lr_dec)
        else:
            optimizer_dec = torch.optim.Adam(list(model.parameters()) + [cluster_centers], lr=args.lr_dec)
    for epoch in range(ee, args.epochs + 1):
        print(epoch)
        model.train()
        # pretraining
        if epoch < args.e:
            recovered, mu, logvar, z, semi_out, _, _ = model(features, adj_norm)
            if args.arga == 1:
                modeldis.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if args.gpu != -1:
                        z_real_dist = z_real_dist.cuda()
                    hidden_emb = mu
                    d_real = modeldis(z_real_dist)
                    d_fake = modeldis(hidden_emb)
                    dis_loss_ = dis_loss(d_real, d_fake)
                    optimizer_dis.zero_grad()
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis.step()
            loss = loss_function(preds=model.dc(z), labels=adj_label, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight,
                                 mu=mu, logvar=logvar, )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hidden_emb = mu.cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Pretrained model result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            model.eval()
            recovered, mu, logvar, z, semi_out, _, _ = model(features, adj_norm)
            hidden_emb = mu.cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Eval model result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            if epoch == args.epoch and args.save_ckpt == 1:
                torch.save(model.state_dict(), f"./pretrain/{args.model}_{args.ds}_{epoch}.pkl")

            if epoch == args.e - 1:
                if args.gpu != -1:
                    cluster_centers = Variable(
                        (torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                        requires_grad=True)
                else:
                    cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)),
                                               requires_grad=True)
                if args.optimi == 'SGD':
                    optimizer_dec = torch.optim.SGD(list(model.parameters()) + [cluster_centers], lr=args.lr_dec)
                else:
                    optimizer_dec = torch.optim.Adam(list(model.parameters()) + [cluster_centers], lr=args.lr_dec)
        # train
        else:
            recovered, mu, logvar, z, semi_out, _, _ = model(features, adj_norm)
            hidden_emb = mu
            if args.arga == 1:
                modeldis.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if args.gpu != -1:
                        z_real_dist = z_real_dist.cuda()
                    d_real = modeldis(z_real_dist)
                    d_fake = modeldis(hidden_emb)
                    dis_loss_ = dis_loss(d_real, d_fake)
                    optimizer_dis.zero_grad()
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis.step()
            loss_re = loss_function(preds=model.dc(z), labels=adj_label, n_nodes=n_nodes, norm=norm,
                                    pos_weight=pos_weight, mu=mu, logvar=logvar, )
            loss_pq, p, q = loss_func(hidden_emb, cluster_centers)
            loss = args.w_pq * loss_pq + args.w_re * loss_re
            cluster_pred_score, cluster_pred = dist_2_label(q)
            acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
            accs.append(float('{:.4f}'.format(acc)))
            print('Trained model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            if epoch > args.e1:
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
                loss += args.w_sup * loss_sup
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
                    for x in q[i]:
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
                margins, indices = q.topk(2, dim=1, largest=True, sorted=True)
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
                print('Pseudo-label:', len(list(idx_train)), sacc)

            print('loss: {:.4f} {:.4f} {:.4f}'.format(loss_re, loss_pq, loss_sup))
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()

            # test
            model.eval()
            recovered, mu, logvar, z, semi_out, _, _ = model(features, adj_norm)
            loss, p, q = loss_func(mu, cluster_centers)
            cluster_pred_score, cluster_pred = dist_2_label(q)
            acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
            print('Eval model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

    print(args)
    print('Cluster accuracy:', accs)
    print('Pseudo-label accuracy:', saccs)
    print('Number of pseudo-labels:', pnum)


def loss_func(feat, cluster_centers):
    alpha = 1.0
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    p = p.detach()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p, q


def dist_2_label(q_t):
    maxlabel, label = torch.max(q_t, dim=1)
    return maxlabel, label


def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    init()
    main()
