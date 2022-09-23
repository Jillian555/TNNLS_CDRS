import os
from models import GMI
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
parser.add_argument('--model', type=str, default='gmi', help="Models used")
parser.add_argument('--optimi', type=str, default='SGD', help="Optimizers used:[SGD ADAM]")
parser.add_argument('--all', type=int, default=0, help='whether to use print, 0/1.')
parser.add_argument('--use_ckpt', type=int, default=0, help='whether to use print, 0/1.')
parser.add_argument('--save_ckpt', type=int, default=0, help='whether to use print, 0/1.')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden2', type=int, default=512, help='Number of units in hidden layer 2.')
parser.add_argument('--hid_units', type=int, default=512,
                    help='dim of node embedding (default: 512)')
parser.add_argument('--negative_num', type=int, default=5,
                    help='number of negative examples used in the discriminator (default: 5)')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='parameter for I(h_i; x_i) (default: 0.8)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='parameter for I(h_i; x_j), node j is a neighbor (default: 1.0)')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='parameter for I(w_ij; a_ij) (default: 1.0)')
parser.add_argument('--activation', default='prelu',
                    help='activation function')
args = parser.parse_args()


def main():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    adj_ori, features, labels, _, _, _ = process.load_data(args.ds)
    features, _ = process.preprocess_features(features)
    labels = torch.FloatTensor(labels[np.newaxis])
    labels = torch.argmax(labels[0, :], dim=1)
    cluster_num = labels.max().item() + 1
    n_nodes, feat_dim = features.shape
    adj_ori = process.normalize_adj(adj_ori + sp.eye(adj_ori.shape[0]))
    sparse = True
    pseudo_labels = torch.full(labels.shape, -1, dtype=torch.long)
    idx_train = set()
    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj_ori)
    else:
        adj_ori = (adj_ori + sp.eye(adj_ori.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj_ori = torch.FloatTensor(adj_ori[np.newaxis])

    model = GMI(feat_dim, args.hid_units, cluster_num, args.activation, )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if args.gpu != -1:
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj_ori = adj_ori.cuda()
        labels = labels.cuda()
        pseudo_labels = pseudo_labels.cuda()

    adj_dense = adj_ori.toarray()
    adj_target = adj_dense + np.eye(adj_dense.shape[0])
    adj_row_avg = 1.0 / np.sum(adj_dense, axis=1)
    adj_row_avg[np.isnan(adj_row_avg)] = 0.0
    adj_row_avg[np.isinf(adj_row_avg)] = 0.0
    adj_dense = adj_dense * 1.0
    for i in range(adj_ori.shape[0]):
        adj_dense[i] = adj_dense[i] * adj_row_avg[i]
    adj_ori = sp.csr_matrix(adj_dense, dtype=np.float32)
    loss_sup = 0.0
    accs = []
    saccs = []
    pnum = []

    ee = 0
    if args.use_ckpt == 1:
        model.load_state_dict(torch.load(f'./pretrain/{args.model}_{args.ds}_{args.epoch}.pkl'))
        ee = args.e
        model.eval()
        res = model(features, adj_ori, args.negative_num, sp_adj, None, None)
        hidden_emb = torch.squeeze(res[5]).cpu().data
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
            res = model(features, adj_ori, args.negative_num, sp_adj, None, None)
            loss = args.alpha * process.mi_loss_jsd(res[0], res[1]) + args.beta * process.mi_loss_jsd(res[2], res[
                3]) + args.gamma * process.reconstruct_loss(res[4], adj_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hidden_emb = torch.squeeze(res[5]).cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Pretrained model result: {:.2f}   {:.2f}   {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            model.eval()
            res = model(features, adj_ori, args.negative_num, sp_adj, None, None)
            hidden_emb = torch.squeeze(res[5]).cpu().data
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
            res = model(features, adj_ori, args.negative_num, sp_adj, None, None)
            loss_re = args.alpha * process.mi_loss_jsd(res[0], res[1]) + args.beta * process.mi_loss_jsd(res[2], res[
                3]) + args.gamma * process.reconstruct_loss(res[4], adj_target)
            hidden_emb = torch.squeeze(res[5])
            semi_out = F.log_softmax(torch.squeeze(res[6]), dim=1)
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

            print('loss: {:.4f} {:.4f} {:.4f}'.format(loss_re, loss_pq, loss_sup))
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()

            model.eval()
            res = model(features, adj_ori, args.negative_num, sp_adj, None, None)
            loss_pq, p, q = loss_func(res[5].squeeze(), cluster_centers)
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
