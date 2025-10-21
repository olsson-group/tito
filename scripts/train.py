import argparse
from datetime import timedelta
import os

import numpy as np
import torch
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader as GeometricDataLoader


from tito import DEVICE
from tito.utils.data import get_dataset
import tito.models.model as model
import tito.models.velocity as velocity
from tito.mlops import get_wandb_logger, get_profiler
import tito.mlops as mlops 

def train_model(args):
    """
    Placeholder function for training a model.
    Replace this with your actual training logic.
    """
    torch.set_float32_matmul_precision('medium')
    print(f"Training model with the following parameters:")
    print(f"Data set: {args.data_set}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    if args.multigpu:
        print(
            "training in multi-gpu mode"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if DEVICE == "cpu":
        num_workers = 0
    else:
        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = len(os.sched_getaffinity(0))
    persistent_workers = True if num_workers > 0 else False

    strategy = DDPStrategy(find_unused_parameters=True) if args.multigpu else "auto"

    profiler = get_profiler(args)

    train_dataset, val_dataset = get_dataset(args)

    #NOTE: we are doing ot in the get_item and not in the collate function now. Collate function not used here therefore

    train_dataloader = GeometricDataLoader(
        train_dataset, batch_size = args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True if num_workers>0 else False, persistent_workers=persistent_workers,
    )
    
    val_dataloader = GeometricDataLoader(
        val_dataset, batch_size = args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if num_workers>0 else False, persistent_workers=persistent_workers,
    )
    
    #instantiate the model
    if hasattr(args, 'from_checkpoint_id') and args.from_checkpoint_id is not None:
        print(f"Loading model from checkpoint {args.from_checkpoint_id} ...")
        project = args.data_set + "-tito"
        ckpt = mlops.get_checkpoint(project, args.from_checkpoint_id, tag=args.checkpoint_tag)
        cfm = model.CFM.load_from_checkpoint(checkpoint_path=ckpt)
    else:
        print("Creating new model ...")
        vf = velocity.PainnCondVelocity(n_features=args.n_features, model_layers=args.n_model_layers, 
                                        embedding_layers=args.n_embedding_layers, length_scale=args.length_scale,
                                        n_reduced_features=args.n_reduced_features, max_lag=args.max_lag)
        cfm = model.CFM(vf, lr=args.learning_rate)

    wandblogger = get_wandb_logger(args, num_workers=num_workers)

    monitor = "val/loss" if not args.no_evaluate else "train/loss"
    model_callback = pl.pytorch.callbacks.ModelCheckpoint(monitor=monitor, 
                                                        filename=f"{args.data_set}-{{epoch:02d}}-{{step}}",
                                                        train_time_interval=timedelta(minutes=args.save_freq))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=DEVICE,
        logger=wandblogger,
        strategy=strategy,
        #precision="bf16" if DEVICE == "gpu" else 32,
        #auto_select_gpus=multigpu,
        devices="auto",#4 if args.multigpu else 1,
        callbacks=[model_callback],
        
        #profiler=profiler,
    )
    if args.no_evaluate:
        trainer.fit(cfm, train_dataloader)
    else:
        trainer.fit(cfm, train_dataloader, val_dataloader)

def main():
    parser = argparse.ArgumentParser(description="TITO training script.")
    parser.add_argument('--data_set', type=str, default="mdqm9", required=False, help="Dataset options: [ala2, mdqm9, timewarp]")
    parser.add_argument("--data_path", type=str, required=False, help="Path to dataset directory.")
    parser.add_argument('--sub_sampling_strategy', type=str, default="None", required=False, help="Sub-sampling strategy: [none, indices]")
    parser.add_argument('--sub_sampling_train_indices', type=str, default=None, required=False, help="Path to sub-sampling indices (.npy file=).")
    parser.add_argument('--sub_sampling_val_indices', type=str, default=None, required=False, help="Path to sub-sampling indices (.npy file).")
    parser.add_argument('--epochs', type=int, default=100000, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--no_evaluate', action='store_true', help="If set, do not run validation.")

    parser.add_argument('--from_checkpoint_id', type=str, help="Wandb run id of checkpoint to start training from.")
    parser.add_argument('--checkpoint_tag', type=str, default="latest", help="Wandb tag of checkpoint to start training from.")
    parser.add_argument('--n_features', type=int, default=64, help="Number of hidden features.")
    parser.add_argument('--n_model_layers', type=int, default=5, help="Number of score layers.")
    parser.add_argument('--n_embedding_layers', type=int, default=2, help="Number of embedding layers.")
    parser.add_argument('--n_reduced_features', type=int, default=0, help="Number of reduced features.")
    parser.add_argument('--max_lag', type=float, default=100., help="Maximum lag for the dataset in ps.") #1000 mdqm9, 2000 timewarp
    parser.add_argument('--length_scale', type=float, default=10.0, help="Length scale for the model.")
    parser.add_argument('--distinguish_atoms', action='store_true', help="If set, distinguish atoms in the dataset.")
    parser.add_argument('--no_uniform_lag', action='store_true', help="Do not use uniform lag (default: False).")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.") #3072 mdqm9,128 for timewarp
    parser.add_argument('--seed', type=int, default=808313, help="Random seed for reproducibility.")
    parser.add_argument('--multigpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument("--save_freq", type=int, default=30, help="Frequency of saving the model in minutes.")
    parser.add_argument('--num_workers', type=int, help="Number workers (GPU Training only).")
    parser.add_argument('--no_ot', action='store_true', help='Disable optimal transport')

    args = parser.parse_args()
    args.mode = "train"  #for compatibility with the dataset loading function

    train_model(args)
if __name__ == "__main__":
    main()
