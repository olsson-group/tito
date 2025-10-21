import matplotlib.pyplot as plt
import torch


def plot_parameters(model):
    ps = []
    for p in model.parameters():
        ps.append(p.flatten())
    ps = torch.cat(ps)
    plot_hist(ps)


def plot_hist(tensor, log=True):
    tensor = tensor.detach().numpy().flatten()
    plt.hist(tensor, bins=100, log=log)
    plt.show()
