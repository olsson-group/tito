# # pylint: disable=not-callable

import numpy as np
import torch

from tito.models import device


class MLP(device.Module):
    def __init__(self, f_in, f_hidden, f_out, skip=False):
        super().__init__()

        self.skip = skip
        self.f_out = f_out

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        if self.skip:
            return x[:, : self.f_out] + self.mlp(x)
        return self.mlp(x)


class InvariantFeatures(device.Module):
    """
    Implement embedding in child class
    All features that will be embedded should be in the batch
    """

    def __init__(self, feature_name, feature_type="node"):
        super().__init__()
        self.feature_name = feature_name
        self.feature_type = feature_type

    def forward(self, batch):
        batch = batch.clone()
        embedded_features = self.embedding(batch[self.feature_name])

        name = f"invariant_{self.feature_type}_features"
        if hasattr(batch, name):
            batch[name] = torch.cat([batch[name], embedded_features], dim=-1)
        else:
            batch[name] = embedded_features

        return batch


class NominalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, n_types, feature_type="node"):
        super().__init__(feature_name, feature_type)
        self.embedding = torch.nn.Embedding(n_types, n_features)


class PositionalEncoder(device.DeviceTracker):
    def __init__(self, dim, max_length):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)]
        encodings = torch.cat(
            encodings,
            axis=1,
        )
        return encodings

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.max_length * rank * np.pi)
        cos = torch.cos(x / self.max_length * rank * np.pi)
        assert cos.device == self.device, f"batch device {cos.device} != model device {self.device}"
        return torch.stack((cos, sin), axis=1)


class PositionalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, length):
        super().__init__(feature_name)
        assert n_features % 2 == 0, "n_features must be even"
        self.rank = n_features // 2
        self.embedding = PositionalEncoder(n_features, length)


class CombineInvariantFeatures(device.Module):
    def __init__(self, n_features_in, n_features_out, skip=False):
        super().__init__()
        self.n_features_out = n_features_out
        self.mlp = MLP(f_in=n_features_in, f_hidden=n_features_out, f_out=n_features_out, skip=skip)

    def forward(self, batch):
        invariant_node_features = self.mlp(batch.invariant_node_features)
        batch.invariant_node_features = invariant_node_features
        return batch


class EdgeEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="edge_type", n_features=n_features, n_types=13, feature_type="edge")


class NodeEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="node_type", n_features=n_features, n_types=167, feature_type="node")


class AddEquivariantFeatures(device.DeviceTracker):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        eq_features = torch.zeros(
            batch.node_type.shape[0],
            self.n_features,
            3,
        )
        batch.equivariant_node_features = eq_features.to(self.device)
        return batch


class EmbedGraph(device.Module):
    def __init__(self, n_features):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            NodeEmbedding(n_features=n_features),
            EdgeEmbedding(n_features=n_features),
            AddEquivariantFeatures(n_features=n_features),
        )

    def forward(self, batch):
        return self.embedding(batch)
