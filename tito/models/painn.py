import warnings

import torch
from torch_scatter import scatter

from tito.models import device, embedding


class Painn(device.Module):
    def __init__(
        self,
        n_features=128,
        n_features_in=None,
        n_features_out=None,
        n_layers=5,
        length_scale=10.0,
        skip=False,
        n_reduced_features=0,
        #attention=False,
    ):
        n_features_in = n_features_in or n_features
        n_features_out = n_features_out or n_features

        super().__init__()
        layers = []

        for l in range(n_layers):
            layers.append(
                SE3Message(
                    n_features=n_features_in if l == 0 else n_features,
                    length_scale=length_scale,
                    n_reduced_features=n_reduced_features,
                    #attention=attention,
                )
            )
            layers.append(Update(n_features))

        layers.append(
            Readout(
                n_features,
                n_features_out,
                skip=skip,
            )
        )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)


class Message(device.Module):
    def __init__(
        self,
        n_features=64,
        length_scale=10.0,
        skip=False,
    ):
        super().__init__()
        self.n_features = n_features

        self.positional_encoder = embedding.PositionalEncoder(n_features, max_length=length_scale)

        phi_in_features = 2 * n_features
        self.phi = embedding.MLP(phi_in_features, n_features, 4 * n_features, skip=skip)
        self.w = embedding.MLP(n_features, n_features, 4 * n_features, skip=skip)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = torch.cat(
            [
                batch.invariant_node_features[src_node],
                batch.invariant_edge_features,
            ],
            dim=-1,
        )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        gates, scale_edge_dir, ds, de = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(gates, batch.equivariant_node_features[src_node])
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dv = scaled_edge_dir + gated_features
        dv = scatter(dv, dst_node, dim=0, reduce="mean")
        ds = scatter(ds, dst_node, dim=0, reduce="mean")

        batch.equivariant_node_features = batch.equivariant_node_features + dv
        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.invariant_edge_features = batch.invariant_edge_features + de

        return batch


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(device.Module):
    def __init__(self, n_features=128, skip=False):
        super().__init__()
        self.u = EquivariantLinear(n_features, n_features)
        self.v = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = embedding.MLP(2 * n_features, n_features, 3 * n_features, skip=skip)

    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        vv = self.v(v)
        uv = self.u(v)

        vv_norm = vv.norm(dim=-1)
        vv_squared_norm = vv_norm**2

        mlp_in = torch.cat([vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        dv = multiply_first_dim(uv, gates)
        ds = vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.equivariant_node_features = batch.equivariant_node_features + dv

        return batch


class EquivariantLinear(device.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class Readout(device.Module):
    def __init__(self, n_features=128, n_features_out=13, skip=False):
        super().__init__()
        self.mlp = embedding.MLP(n_features, n_features, 2 * n_features_out, skip=skip)
        self.V = EquivariantLinear(n_features, n_features_out)  # pylint:disable=invalid-name
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(equivariant_node_features, gates)

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch


class Reducer(device.Module):
    def __init__(self, n_features, n_reduced_features):
        super().__init__()
        self.reduce_invariant_node_features = torch.nn.Linear(n_features, n_reduced_features)
        self.reduce_invariant_edge_features = torch.nn.Linear(n_features, n_reduced_features)
        self.reduce_equivariant_node_features = EquivariantLinear(n_features, n_reduced_features)

        self.expand_equivariant_features = EquivariantLinear(n_reduced_features, n_features)
        self.expand_invariant_features = torch.nn.Linear(n_reduced_features, n_features)
        self.expand_invariant_edges_features = torch.nn.Linear(n_reduced_features, n_features)


class SE3Message(device.Module):
    def __init__(self, n_features=64, n_reduced_features=0, length_scale=10.0):#, attention=False):
        super().__init__()
        self.n_features = n_features
        self.n_reduced_features = n_reduced_features
        #self.attention = attention

        n_hidden_features = n_features
        if n_reduced_features > 0:
            self.reducer = Reducer(n_features, n_reduced_features)
            n_hidden_features = n_reduced_features

        self.positional_encoder = embedding.PositionalEncoder(n_hidden_features, max_length=length_scale)
        self.phi = embedding.MLP(2 * n_hidden_features, n_hidden_features, 5 * n_hidden_features)
        self.w = embedding.MLP(n_hidden_features, n_hidden_features, 5 * n_hidden_features)
        self.n_hidden_features = n_hidden_features

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        invariant_node_features = batch.invariant_node_features
        invariant_edge_features = batch.invariant_edge_features
        equivariant_node_features = batch.equivariant_node_features

        if self.n_reduced_features > 0:
            invariant_node_features = self.reducer.reduce_invariant_node_features(invariant_node_features)
            invariant_edge_features = self.reducer.reduce_invariant_edge_features(invariant_edge_features)
            equivariant_node_features = self.reducer.reduce_equivariant_node_features(
                equivariant_node_features
            )

        in_features = torch.cat(
            [
                invariant_node_features[src_node],
                invariant_edge_features,
            ],
            dim=-1,
        )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        invariant_features = self.phi(in_features) * self.w(positional_encoding)
        gates, scale_edge_dir, ds, de, cross_product_gates = torch.split(
            invariant_features,
            int(len(invariant_features[0]) / 5),
            dim=-1,
        )

        gated_features = multiply_first_dim(gates, equivariant_node_features[src_node])
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_hidden_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_hidden_features, 1)
        src_equivariant_node_features = equivariant_node_features[dst_node]
        cross_produts = torch.cross(dst_node_edges, src_equivariant_node_features, dim=-1)

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products

        # if self.attention:
        #     attention_scores = composite.scatter_softmax(ds[:, 0], dst_node)
        #     dv = attention_scores.view(-1, 1, 1) * dv
        #     ds = attention_scores.view(-1, 1) * ds

        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        if self.n_reduced_features > 0:
            dv = self.reducer.expand_equivariant_features(dv)
            ds = self.reducer.expand_invariant_features(ds)
            de = self.reducer.expand_invariant_edges_features(de)

        batch.equivariant_node_features = batch.equivariant_node_features + dv
        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.invariant_edge_features = batch.invariant_edge_features + de

        return batch
