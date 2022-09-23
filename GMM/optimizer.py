import torch.nn.functional as F


def loss_function(preds, labels, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    return cost
