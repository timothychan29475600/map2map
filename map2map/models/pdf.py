import torch 

def prob_den(x,no_bins=201,eps=1e-8):
    """ Compute pdf of the given fields """
    data = x + eps
    dtmin, dtmax = torch.min(data), torch.max(data)
    deltaX = (dtmax-dtmin)/no_bins

    if torch.isinf(dtmax) or torch.isinf(dtmin):
        bins_edges = torch.linspace(0, 0.1, no_bins+1)
        bins = 0.5*(bins_edges[:-1] + bins_edges[1:])
        hist = torch.zeros_like(bins)
        deltaX = bins_edges[1]-bins_edges[0]
        return bins, hist, deltaX

    bins_edges = torch.linspace(dtmin, dtmax, no_bins+1)
    bins = 0.5*(bins_edges[:-1] + bins_edges[1:])

    hist = torch.histc(data,bins=no_bins,min=dtmin,max=dtmax)
    print('Histogram ', hist.shape)
    print('Bins ',bins.shape)
    hist /= hist.sum()
    hist /= deltaX
    return bins, hist, deltaX


