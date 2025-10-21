from setuptools import find_packages, setup

setup(
    name="tito",
    packages=find_packages(),
    install_requires=[
        #"torch", #commened out torch-related dependencies to control torch and cuda versions
        #"lightning",
        #"torch_geometric",
        #"torchdyn",
        "mdtraj",
        "tqdm",
        "mdshare",
        "deeptime",
        "matplotlib",
        "h5py",
        "wandb",
        "rdkit",
        "scipy",
    ],
)
