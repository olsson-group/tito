import torch
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl
import copy
import weakref
from scipy.optimize import linear_sum_assignment

from pdb import set_trace as st

def create_efficient_collate_fn(dataset):
    """
    Factory function that creates an efficient collate function compatible with 
    both GeometricDataLoader and PyTorch Lightning
    """
    
    def collate_fn(batch):
        """
        Efficient collate function that handles the expensive OT operations
        Compatible with GeometricDataLoader and Lightning
        """
        # Extract data from batch
        data0_list = [item["data0"] for item in batch]
        datat_list = [item["datat"] for item in batch]
        lag_list = [item["lag"] for item in batch]
        
        # Process each item sequentially for now (see note below about multiprocessing)
        processed_items = []
        
        for i, (data0, datat, lag) in enumerate(zip(data0_list, datat_list, lag_list)):
            # Generate base sample
            base_sample = dataset.__basedistribution.sample_as(datat.x)
            
            # Perform OT coupling (optimized version)
            base, target = optimized_OT_coupler(base_sample, datat.x, plan="rp")
            
            # Create new data object
            new_datat = copy.deepcopy(datat)
            new_datat.x = target
            new_datat.xbase = base
            
            item = {"cond": data0, "target": new_datat, "lag": torch.tensor([lag])}
            processed_items.append(dataset.transform(item))
        
        # Use GeometricDataLoader's batching mechanism
        cond_batch = Batch.from_data_list([item["cond"] for item in processed_items])
        target_batch = Batch.from_data_list([item["target"] for item in processed_items])
        lag_batch = torch.cat([item["lag"] for item in processed_items])

        print (f"target_batch keys: {target_batch.keys()}")
        return {"cond": cond_batch, "target": target_batch, "lag": lag_batch}
    
    return collate_fn

def optimized_OT_coupler(base, target, plan="rp"):

    plan_map = {
        "r": optimized_rot_ot,
        "p": optimized_permute_ot
    }
    
    for c in plan:
        base, target = plan_map[c](base, target)
    
    return base, target

def optimized_rot_ot(base, target):
    H = torch.matmul(base.T, target)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.matmul(Vt.T, U.T))
    if d < 0.0: # check if rotation is improper
        Vt[-1] *= -1.0
    R = torch.matmul(Vt.T, U.T)
    base_rotated = torch.matmul(base, R.T)
    return base_rotated, target

def optimized_permute_ot(base, target):
    # Use GPU for distance calculation if available and data is large enough
    if torch.cuda.is_available() and base.shape[0] > 100:
        with torch.no_grad():
            base_gpu = base.cuda()
            target_gpu = target.cuda()
            cost_matrix = torch.cdist(base_gpu, target_gpu, p=2).T.cpu().numpy()
    else:
        cost_matrix = torch.cdist(base, target, p=2).T.numpy()
    
    # Use scipy linear_sum_assignment (fastest in benchmarks)
    row_indices, _ = linear_sum_assignment(cost_matrix)
    
    # Apply permutation
    base_permuted = base[row_indices]
    
    return base_permuted, target

def create_threaded_collate_fn(dataset, max_workers=4):
    """
    Factory is used to avoid overhead of creating ThreadPoolExecutor for each batch.
    This function creates a collate function that processes OT coupling in parallel using threads.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def collate_fn(batch):
        # TODO: consider adding minibatch OT as well if overhead is not too high.
        # Extract data from batch
        data0_list = [item["data0"] for item in batch]
        datat_list = [item["datat"] for item in batch]
        lag_list = [item["lag"] for item in batch]
        
        # Process OT coupling in parallel using threads
        def process_single_ot(data0, datat, lag):
            base_sample = dataset.__basedistribution.sample_as(datat.x)
            base, target = optimized_OT_coupler(base_sample, datat.x, plan="rp")
            
            new_datat = copy.deepcopy(datat)
            new_datat.x = target
            new_datat.xbase = base
            
            item = {"cond": data0, "target": new_datat, "lag": torch.tensor([lag])}
            return dataset.transform(item)
        
        # Submit all tasks to thread pool
        futures = [
            executor.submit(process_single_ot, data0, datat, lag)
            for data0, datat, lag in zip(data0_list, datat_list, lag_list)
        ]
        
        # Collect results
        processed_items = [future.result() for future in futures]
        
        # Batch the processed items
        cond_batch = Batch.from_data_list([item["cond"] for item in processed_items])
        target_batch = Batch.from_data_list([item["target"] for item in processed_items])
        lag_batch = torch.cat([item["lag"] for item in processed_items])
        
        return {"cond": cond_batch, "target": target_batch, "lag": lag_batch}
    
    # cleanup
    weakref.finalize(collate_fn, lambda e=executor: e.shutdown())
    
    return collate_fn
