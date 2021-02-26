import torch




def mid_p_value(idx, dist):
    """
    Computes the discrete mid-p-value of each item from idx using
    the discrete distribution dist.

    Arguments
    ---------
    idx : torch.LongTensor
        Indices of the items whose p-value should be computed
        (shape: (n_items,)).
    dist : torch.Tensor
        Discrete distribution used to compute p-values (shape:
        (n_items, set_size)).
        Each line of this tensor represents a probability distribution
        over the set from which the items are sampled.

    Returns
    -------
    p_values : torch.Tensor
        Discrete mid-p-values of the items (shape: (n_items,)).
    """
    tmp = list(range(len(idx)))
    inf_pval = (dist*(dist < dist[tmp, idx].unsqueeze(1))).sum(1)
    sup_pval = (dist*(dist <= dist[tmp, idx].unsqueeze(1))).sum(1)
    return .5*(inf_pval + sup_pval)
