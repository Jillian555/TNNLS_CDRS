import os
import torch.nn as nn
from models import DGI
from utils import process
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import KMeans
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
parser.add_argument('--lr_dec', type=float, default=0.15, help='Learning rate.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--e', type=int, default=100, help='Number of pretrained epochs.')
parser.add_argument('--e1', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e2', type=int, default=200, help='Number of epochs.')
parser.add_argument('--e3', type=int, default=300, help='Number of epochs.')
parser.add_argument('--epoch', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--update', type=int, default=10, help='update epoch.')
parser.add_argument('--model', type=str, default='dgi', help="Models used")
parser.add_argument('--optimi', type=str, default='SGD', help="Optimizers used:[SGD ADAM]")
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden2', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--save_ckpt', type=int, default=0, help='whether to save checkpoint, 0/1.')
parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use checkpoint, 0/1.')
args = parser.parse_args()


def main():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    adj, features, labels, _, _, _ = process.load_data(args.ds)
    features, _ = process.preprocess_features(features)
    labels = torch.FloatTensor(labels[np.newaxis])
    labels = torch.argmax(labels[0, :], dim=1)
    cluster_num = labels.max().item() + 1
    n_nodes, feat_dim = features.shape
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    nonlinearity = 'prelu'  # special name to separate parameters
    batch_size = 1
    sparse = True
    pseudo_labels = torch.full(labels.shape, -1, dtype=torch.long)
    idx_train = set()
    b_xent = nn.BCEWithLogitsLoss()
    loss_sup = 0.0
    accs = []
    saccs = []
    pnum = []

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    idx = np.random.permutation(n_nodes)
    shuf_fts = features[:, idx, :]
    #fix shuf_fts
    shuf_fts = torch.load('shuf_fts_{}.pkl'.format(args.ds), map_location='cpu')
    model = DGI(feat_dim, args.hidden1, args.hidden2, cluster_num, nonlinearity)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if args.gpu != -1:
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()
        labels = labels.cuda()
        pseudo_labels = pseudo_labels.cuda()
        shuf_fts = shuf_fts.cuda()

    ee = 0
    if args.use_ckpt == 1:
        ee = args.e
        model.load_state_dict(torch.load(f'./pretrain/{args.model}_{args.ds}_{args.epoch}.pkl'))
        model.eval()
        logits, embeds, output = model(features, shuf_fts, sp_adj if sparse else adj, None, None, None)
        hidden_emb = torch.squeeze(embeds).cpu().data
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
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
    for epoch in range(ee, args.epochs):
        print(epoch)
        model.train()
        if epoch < args.e:
            lbl_1 = torch.ones(batch_size, n_nodes)
            lbl_2 = torch.zeros(batch_size, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if args.gpu != -1:
                lbl = lbl.cuda()

            logits, embeds, output = model(features, shuf_fts, sp_adj if sparse else adj, None, None, None)
            loss = b_xent(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hidden_emb = torch.squeeze(embeds).cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Pretrained model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            model.eval()
            logits, embeds, output = model(features, shuf_fts, sp_adj if sparse else adj, None, None, None)
            hidden_emb = torch.squeeze(embeds).cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Eval model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
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
        else:
            lbl_1 = torch.ones(batch_size, n_nodes)
            lbl_2 = torch.zeros(batch_size, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if args.gpu != -1:
                lbl = lbl.cuda()
            logits, embeds, output = model(features, shuf_fts, sp_adj if sparse else adj, None, None, None)
            hidden_emb = torch.squeeze(embeds)
            semi_out = F.log_softmax(torch.squeeze(output), dim=1)
            loss_re = b_xent(logits, lbl)
            loss_pq, p, q = loss_func(hidden_emb, cluster_centers)
            loss = args.w_re * loss_re + args.w_pq * loss_pq
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
            if epoch % args.update == 0 and epoch > args.e2:
                pnum.append(len(list(idx_train)))
                sacc, _, _ = cluster_accuracy(pseudo_labels[idx_train].cpu(), labels[idx_train].cpu(), cluster_num)
                saccs.append(float('{:.4f}'.format(sacc)))
                print('pseudo', len(list(idx_train)), sacc)


            print('loss: {:.4f} {:.8f} {:.4f}'.format(loss_re, loss_pq, loss_sup))
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()

            model.eval()
            logits, embeds, output = model(features, shuf_fts, sp_adj if sparse else adj, None, None, None)
            hidden_emb = embeds.squeeze()
            loss_pq, p, q = loss_func(hidden_emb, cluster_centers)
            cluster_pred_score, cluster_pred = dist_2_label(q)
            acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
            print('Eval result: {:.2f} {:.2f} {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

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
