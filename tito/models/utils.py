from datetime import datetime

import mdtraj as md
import torch_geometric as geom
import torch
    

def _get_COMs(x, batch):
    num_points = torch.bincount(batch)
    sum_coords = torch.zeros(len(num_points), x.shape[1], device=x.device)
    #cpu cuda:0 cpu
#    print(x.device, batch.device, sum_coords.device)
    sum_coords.index_add_(0, batch, x)  # Sum coordinates for each batch member
    com = sum_coords / num_points[:, None]  # Compute center of mass for each batch member
    return com
    
def all_centered(x, batch):
    coms = _get_COMs(x, batch)
    num_points = torch.bincount(batch)
    return torch.all(torch.isclose(coms, torch.zeros((len(num_points), x.shape[1]),device=coms.device), atol=1e-5, rtol=1e-5)).item()


def center_coordinates_batch(x, batch):
    # Center batch
    com = _get_COMs(x, batch)
    expanded_com = com[batch]
    return x - expanded_com

def center_coordinates(x):
    com = x.mean(dim=0, keepdim=True)
    return x - com

class Timer:
    def __init__(self):
        self.start = datetime.now()

    def __str__(self):
        now = datetime.now()
        time_passed = now - self.start
        return str(time_passed).split(".")[0]


def get_example_batch(dataset, batch_size):
    example_batch = next(iter(geom.loader.DataLoader(dataset, batch_size=batch_size)))
    return example_batch

