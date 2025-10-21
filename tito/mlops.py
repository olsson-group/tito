import os
import pickle as pkl
import wandb
import shutil
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler



def save(name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pkl.dump(name, f)


def load(path):
    with open(path, "rb") as f:
        return pkl.load(f)
    
def get_artifact(project, run_id, tag="best"):
    api = wandb.Api()
    artifact = api.artifact(
        os.path.join(project, f"model-{run_id}:{tag}"), type="model"
    )
    #wandb.init(project=project)
    #artifact_path = f"juan-viguera/{project}/model-{run_id}:{tag}"
    #artifact = wandb.use_artifact(artifact_path, type='model')
    return artifact

def get_checkpoint(project, run_id, tag="best"):
    artifact_dir = get_artifact(project, run_id, tag=tag).download()
    artifact_dir = fix_artifact_dir(artifact_dir)
    return os.path.join(artifact_dir, "model.ckpt")

def fix_artifact_dir(artifact_dir):
    new_path = artifact_dir.replace(":", "_")
    if not os.path.exists(new_path):
        shutil.move(artifact_dir, new_path)
    artifact_dir = new_path
    return artifact_dir

def get_wandb_logger(args, num_workers=0):
    wandblogger = WandbLogger(
            project=f"{args.data_set}-tito",
            config={
                "data_set": args.data_set,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "n_features": args.n_features,
                "n_model_layers": args.n_model_layers,
                "n_embedding_layers": args.n_embedding_layers,
                "max_lag": args.max_lag,
                "length_scale": args.length_scale,
                "n_reduced_features": args.n_reduced_features,
                "no_uniform_lag": args.no_uniform_lag,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "multigpu": args.multigpu,
                "num_workers": num_workers,
                "distinguish_atoms": args.distinguish_atoms,
            },
            log_model="all",
        )
    return wandblogger

def get_profiler(args):
    profiler = PyTorchProfiler(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
)
    return profiler