import torch
import torch_geometric as geom
from torch_scatter import scatter

from tito import utils
from tito.models import device, graph, painn , embedding 
#from tito.models import old_embeddings
from tito.utils import timer
from tito.models.utils import all_centered, center_coordinates_batch

class PainnCondVelocity(device.Module):
    def __init__(
        self,
        n_features=64,
        cutoff=None,
        virtual_node=False,
        embedding_layers=2,
        model_layers=5,
        max_lag=100,
        length_scale=10.0,
        n_reduced_features=0,
        temperature=False,
    ):
        super().__init__()
        self.config = {"n_features": n_features, 
                       "cutoff": cutoff, 
                       "virtual_node": virtual_node,
                       "embedding_layers": embedding_layers,
                       "model_layers": model_layers,
                       "max_lag": max_lag,
                       "length_scale": length_scale,
                       "n_reduced_features": n_reduced_features,
                       "temperature": temperature,
                       }

        self.cutoff = cutoff
        self.virtual_node = virtual_node

        self.temperature = temperature
        self.embed = torch.nn.Sequential(
            graph.AddSpatialFeatures(),
            embedding.NodeEmbedding(n_features=n_features),
            embedding.EdgeEmbedding(n_features=n_features),
            embedding.AddEquivariantFeatures(n_features=n_features),
            painn.Painn(
                n_features=n_features,
                n_layers=embedding_layers,
                length_scale=length_scale,
                n_reduced_features=n_reduced_features,
            ),
        )

        n_invariant_features = 3 + (1 if temperature else 0)
        self.score = torch.nn.Sequential(
            graph.AddSpatialFeatures(),
            embedding.PositionalEmbedding("t_diff", n_features, 1),
            (
                embedding.PositionalEmbedding("temperature", n_features, 1)
                if temperature
                else torch.nn.Identity()
            ),
            embedding.PositionalEmbedding("lag", n_features, max_lag),
            embedding.CombineInvariantFeatures(n_invariant_features * n_features, n_features),
            painn.Painn(
                n_features=n_features,
                n_features_out=1,
                n_layers=model_layers,
                n_reduced_features=n_reduced_features,
            ),
        )

        self.radius_edges = graph.AddRadiusGraph(cutoff=self.cutoff)
        self.bond_edges = graph.AddBondGraph()

        if self.virtual_node:
            self.add_virtual_node = graph.AddVirtualNode()
            self.virtual_edges = graph.AddVirtualGraph()

        self.timer = timer.Timer()

    def preprocess(self, batch):
        if self.virtual_node:
            #  assert utils.is_centered(batch["cond"])
            #  assert utils.is_centered(batch["target"])

            batch["cond"] = self.add_virtual_node(batch["cond"])
            batch["target"] = self.add_virtual_node(batch["target"])

        return batch

    def forward(self, t, batch, logger=None):
        corr = batch["corr"].clone()
        cond = batch["cond"].clone()

        edge_index, edge_type = self.get_edge_index(corr, cond) 

        batch_idx = cond.batch
        corr.lag = batch["lag"][batch_idx].squeeze()
        corr.t_diff = t[batch["cond"].batch] #batch["t_diff"][batch_idx].squeeze()

        if not hasattr(self, "temperature"):
            self.temperature = False
        if self.temperature:
            corr.temperature = batch["temperature"][batch_idx].squeeze()

        cond.edge_index = edge_index
        cond.edge_type = edge_type
        cond = self.embed(cond)

        corr.edge_index = edge_index
        corr.edge_type = edge_type
        corr.invariant_node_features = cond.invariant_node_features
        corr.invariant_edge_features = cond.invariant_edge_features
        # Initialize with random equivariant features, as OT is not done for the condition only the target
        corr.equivariant_node_features = torch.randn_like(cond.equivariant_node_features) 

        dx = corr.x + self.score(corr).equivariant_node_features.squeeze() 
        dx = center_coordinates_batch(dx, corr.batch) 
        #print('is dx centered', all_centered(dx, corr.batch), flush=True)
        #epsilon_hat = corr.clone()

        #epsilon_hat.x = epsilon_hat.x + dx
        #epsilon_hat = utils.center_batch(epsilon_hat)

        if logger is not None:
            corr_edges = corr.edge_index.shape[1] / corr.node_type.shape[0]
            cond_edges = cond.edge_index.shape[1] / cond.node_type.shape[0]
            logger.log_metrics(
                {
                    "corr_edges": corr_edges,
                    "cond_edges": cond_edges,
                }
            )

        return dx

    def get_edge_index(self, corr, cond):
        cond_radius_edges, cond_radius_edge_type = self.radius_edges.get_edges(cond)
        corr_radius_edges, corr_radius_edge_type = self.radius_edges.get_edges(corr)
        bond_edges, bond_edge_type = self.bond_edges.get_edges(corr)

        edge_index = torch.cat([cond_radius_edges, corr_radius_edges, bond_edges], dim=1)
        edge_type = torch.cat([cond_radius_edge_type, corr_radius_edge_type, bond_edge_type])

        if self.virtual_node:
            virtual_edges, virtual_edge_type = self.virtual_edges.get_edges(corr)
            edge_index = torch.cat([edge_index, virtual_edges], dim=1)
            edge_type = torch.cat([edge_type, virtual_edge_type])

        edge_index, edge_type = geom.utils.coalesce(
            edge_index,
            edge_type,
            reduce="max",
        )

        return edge_index, edge_type


# class PainnUncondVelocity(device.Module):
#     def __init__(
#         self,
#         n_features=64,
#         cutoff=None,
#         virtual_node=False,
#         embedding_layers=2,
#         model_layers=5,
#         max_lag=100,
#         length_scale=10.0,
#         n_reduced_features=0,
#         temperature=False,
#     ):
#         super().__init__()
#         self.config = {"n_features": n_features, 
#                        "cutoff": cutoff, 
#                        "virtual_node": virtual_node,
#                        "embedding_layers": embedding_layers,
#                        "model_layers": model_layers,
#                        "max_lag": max_lag,
#                        "length_scale": length_scale,
#                        "n_reduced_features": n_reduced_features,
#                        "temperature": temperature,
#                        }

#         self.cutoff = cutoff
#         self.virtual_node = virtual_node

#         self.embed = torch.nn.Sequential(
#             graph.AddSpatialFeatures(),
#             embedding.NodeEmbedding(n_features=n_features),
#             embedding.EdgeEmbedding(n_features=n_features),
#             embedding.AddEquivariantFeatures(n_features=n_features),
#             painn.Painn(
#                 n_features=n_features,
#                 n_layers=embedding_layers,
#                 length_scale=length_scale,
#                 n_reduced_features=n_reduced_features,
#             ),
#         )

#         n_invariant_features = 2#3 + (1 if temperature else 0)
#         self.score = torch.nn.Sequential(
#             graph.AddSpatialFeatures(),
#             embedding.PositionalEmbedding("t_diff", n_features, 1),
#             embedding.CombineInvariantFeatures(n_invariant_features * n_features, n_features),
#             painn.Painn(
#                 n_features=n_features,
#                 n_features_out=1,
#                 n_layers=model_layers,
#                 n_reduced_features=n_reduced_features,
#             ),
#         )

#         self.radius_edges = graph.AddRadiusGraph(cutoff=self.cutoff)
#         self.bond_edges = graph.AddBondGraph()

#         if self.virtual_node:
#             self.add_virtual_node = graph.AddVirtualNode()
#             self.virtual_edges = graph.AddVirtualGraph()

#         self.timer = timer.Timer()

#     def preprocess(self, batch):
#         if self.virtual_node:
#             #  assert utils.is_centered(batch["cond"])
#             #  assert utils.is_centered(batch["target"])

#             batch["cond"] = self.add_virtual_node(batch["cond"])
#             batch["target"] = self.add_virtual_node(batch["target"])

#         return batch

#     def forward(self, t, batch, logger=None):
#         corr = batch["corr"].clone()
#         #cond = batch["cond"].clone()

#         edge_index, edge_type = self.get_edge_index(corr, None)

#         batch_idx = corr.batch
#         corr.lag = batch["lag"][batch_idx].squeeze()
#         corr.t_diff = t[batch["corr"].batch] #batch["t_diff"][batch_idx].squeeze()

#         if not hasattr(self, "temperature"):
#             self.temperature = False
#         if self.temperature:
#             corr.temperature = batch["temperature"][batch_idx].squeeze()

#         #cond.edge_index = edge_index
#         #cond.edge_type = edge_type
#         #cond = self.embed(cond)
        
#         corr.edge_index = edge_index
#         corr.edge_type = edge_type
#         corr = self.embed(corr)

#         corr.edge_index = edge_index
#         corr.edge_type = edge_type
#         # corr.invariant_node_features = cond.invariant_node_features
#         # corr.invariant_edge_features = cond.invariant_edge_features
#         # corr.equivariant_node_features = cond.equivariant_node_features

#         dx = self.score(corr).equivariant_node_features.squeeze()

#         #epsilon_hat = corr.clone()

#         #epsilon_hat.x = epsilon_hat.x + dx
#         #epsilon_hat = utils.center_batch(epsilon_hat)

#         if logger is not None:
#             corr_edges = corr.edge_index.shape[1] / corr.node_type.shape[0]
#             #cond_edges = cond.edge_index.shape[1] / cond.node_type.shape[0]
#             logger.log_metrics(
#                 {
#                     "corr_edges": corr_edges,
#                     #"cond_edges": cond_edges,
#                 }
#             )

#         return dx

#     def get_edge_index(self, corr, cond):
#         #cond_radius_edges, cond_radius_edge_type = self.radius_edges.get_edges(cond)
#         corr_radius_edges, corr_radius_edge_type = self.radius_edges.get_edges(corr)
#         bond_edges, bond_edge_type = self.bond_edges.get_edges(corr)
        
#         edge_index = torch.cat([corr_radius_edges, bond_edges], dim=1)
#         edge_type = torch.cat([corr_radius_edge_type, bond_edge_type])

#         #edge_index = torch.cat([cond_radius_edges, corr_radius_edges, bond_edges], dim=1)
#         #edge_type = torch.cat([cond_radius_edge_type, corr_radius_edge_type, bond_edge_type])

#         if self.virtual_node:
#             virtual_edges, virtual_edge_type = self.virtual_edges.get_edges(corr)
#             edge_index = torch.cat([edge_index, virtual_edges], dim=1)
#             edge_type = torch.cat([edge_type, virtual_edge_type])

#         # edge_index, edge_type = geom.utils.coalesce(
#         #     edge_index,
#         #     edge_type,
#         #     reduce="max",
#         # )

#         return edge_index, edge_type

# class PainnScore(device.Module):
#     def __init__(
#         self,
#         n_features=64,
#         cutoff=None,
#         virtual_node=False,
#         model_layers=5,
#         max_lag=100,
#         skip=False,
#         length_scale=10,
#         n_reduced_features=0,
#         temperature=False,
#     ):
#         super().__init__()
#         if n_reduced_features > 0 or temperature:
#             raise NotImplementedError

#         self.cutoff = cutoff
#         self.virtual_node = virtual_node

#         self.score = torch.nn.Sequential(
#             graph.AddSpatialFeatures(),
#             embedding.NodeEmbedding(n_features=n_features),
#             embedding.EdgeEmbedding(n_features=n_features),
#             embedding.AddEquivariantFeatures(n_features=n_features),
#             embedding.PositionalEmbedding("t_diff", n_features, 1),
#             embedding.PositionalEmbedding("lag", n_features, max_lag),
#             embedding.CombineInvariantFeatures(3 * n_features, n_features),
#             painn.Painn(
#                 n_features=n_features,
#                 n_features_out=1,
#                 n_layers=model_layers,
#                 skip=skip,
#                 length_scale=length_scale,
#             ),
#         )

#         self.radius_edges = graph.AddRadiusGraph(cutoff=self.cutoff)
#         self.bond_edges = graph.AddBondGraph()

#         if self.virtual_node:
#             self.add_virtual_node = graph.AddVirtualNode()
#             self.virtual_edges = graph.AddVirtualGraph()

#         self.timer = timer.Timer()

#     def preprocess(self, batch):
#         if self.virtual_node:
#             batch["target"] = self.add_virtual_node(batch["target"])

#         return batch

#     def forward(self, batch, logger=None):
#         breakpoint()
#         corr = batch["corr"].clone()
#         corr = self.preprocess(corr)

#         edge_index, edge_type = self.get_edge_index(corr)

#         batch_idx = corr.batch
#         corr.lag = batch["lag"][batch_idx].squeeze()
#         corr.t_diff = batch["t_diff"][batch_idx].squeeze()

#         corr.edge_index = edge_index
#         corr.edge_type = edge_type

#         dx = self.score(corr).equivariant_node_features.squeeze()

#         corr.x += dx
#         corr = utils.center_batch(corr)

#         if logger is not None:
#             corr_edges = corr.edge_index.shape[1] / corr.node_type.shape[0]
#             logger.log_metrics(
#                 {
#                     "corr_edges": corr_edges,
#                 }
#             )

#         return corr

#     def get_edge_index(self, corr):
#         corr_radius_edges, corr_radius_edge_type = self.radius_edges.get_edges(corr)
#         bond_edges, bond_edge_type = self.bond_edges.get_edges(corr)

#         edge_index = torch.cat([corr_radius_edges, bond_edges], dim=1)
#         edge_type = torch.cat([corr_radius_edge_type, bond_edge_type])

#         if self.virtual_node:
#             virtual_edges, virtual_edge_type = self.virtual_edges.get_edges(corr)
#             edge_index = torch.cat([edge_index, virtual_edges], dim=1)
#             edge_type = torch.cat([edge_type, virtual_edge_type])

#         edge_index, edge_type = geom.utils.coalesce(
#             edge_index,
#             edge_type,
#             reduce="max",
#         )

#         return edge_index, edge_type



# #code from Weilong in TI
# import warnings

# class PaiNNTLVelocity(device.Module):
#     def __init__(
#         self,
#         n_features=64,
#         model_layers=4,
#         max_lag=1000,
#         n_neighbors=1000,
#         n_types=167,
#         dist_encoding="positional_encoding",
#         embedding_layers = None,  #WARNING!: Not used in this class, but kept for compatibility
#         length_scale=None, #WARNING!: Not used in this class, but kept for compatibility
#         n_reduced_features=None, #WARNING!: Not used in this class, but kept for compatibility
#     ):
#         super().__init__()
#         self.embed = torch.nn.Sequential(
#             old_embeddings.AddEdges(n_neighbors=n_neighbors),
#             old_embeddings.AddEquivariantFeatures(n_features),
#             old_embeddings.NominalEmbedding("node_type", n_features, n_types=n_types),
#             old_embeddings.PositionalEmbedding("lag", n_features, max_lag),
#             old_embeddings.CombineInvariantFeatures(2 * n_features, n_features),
#             PaiNNBase(
#                 n_features=n_features,
#                 n_features_out=n_features,
#                 n_layers=model_layers,
#                 dist_encoding=dist_encoding,
#             ),
#         )

#         self.net = torch.nn.Sequential(
#             old_embeddings.AddEdges(should_generate_edge_index=False),
#             old_embeddings.PositionalEmbedding("t_diff", n_features, 1.),
#             old_embeddings.CombineInvariantFeatures(2 * n_features, n_features),
#             PaiNNBase(n_features=n_features, dist_encoding=dist_encoding),
#         )

#     def forward(self, t, batch):

#         batch_0 = batch["cond"].clone()
#         noise_batch = batch["corr"].clone()
        
#         batch_idx = batch_0.batch
#         batch_0.lag = batch["lag"][batch_idx].squeeze()
#         noise_batch.t_diff = t[batch["cond"].batch]

#         embedded = self.embed(batch_0)
#         cond_inv_features = embedded.invariant_node_features
#         cond_eqv_features = embedded.equivariant_node_features
#         cond_edge_index = embedded.edge_index

#         noise_batch.invariant_node_features = cond_inv_features
#         noise_batch.equivariant_node_features = cond_eqv_features
#         noise_batch.edge_index = cond_edge_index

#         dx = self.net(noise_batch).equivariant_node_features.squeeze()
#         noise_batch.x = noise_batch.x + dx

#         return noise_batch.x


# class PaiNNBase(torch.nn.Module):
#     def __init__(
#         self,
#         n_features=128,
#         n_layers=5,
#         n_features_out=1,
#         length_scale=10,
#         dist_encoding="positional_encoding",
#     ):
#         super().__init__()
#         layers = []
#         for _ in range(n_layers):
#             layers.append(
#                 Message(
#                     n_features=n_features,
#                     length_scale=length_scale,
#                     dist_encoding=dist_encoding,
#                 )
#             )
#             layers.append(Update(n_features))

#         layers.append(Readout(n_features, n_features_out))
#         self.layers = torch.nn.Sequential(*layers)

#     def forward(self, batch):
#         return self.layers(batch)


# class Message(torch.nn.Module):
#     def __init__(
#         self, n_features=128, length_scale=10, dist_encoding="positional_encoding"
#     ):
#         super().__init__()
#         self.n_features = n_features

#         assert dist_encoding in (
#             a := ["positional_encoding", "soft_one_hot"]
#         ), f"positional_encoder must be one of {a}"

#         if dist_encoding in ["positional_encoding", None]:
#             self.positional_encoder = old_embeddings.PositionalEncoder(
#                 n_features, length=length_scale
#             )
#         elif dist_encoding == "soft_one_hot":
#             self.positional_encoder = old_embeddings.SoftOneHotEncoder(
#                 n_features, max_radius=length_scale
#             )

#         self.phi = old_embeddings.MLP(n_features, n_features, 4 * n_features)
#         self.W = old_embeddings.MLP(n_features, n_features, 4 * n_features)

#     def forward(self, batch):
#         src_node = batch.edge_index[0]
#         dst_node = batch.edge_index[1]

#         positional_encoding = self.positional_encoder(batch.edge_dist)
#         gates, cross_product_gates, scale_edge_dir, scale_features = torch.split(
#             self.phi(batch.invariant_node_features[src_node])
#             * self.W(positional_encoding),
#             self.n_features,
#             dim=-1,
#         )
#         gated_features = multiply_first_dim(
#             gates, batch.equivariant_node_features[src_node]
#         )
#         scaled_edge_dir = multiply_first_dim(
#             scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
#         )

#         dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
#         dst_equivariant_node_features = batch.equivariant_node_features[dst_node]
#         cross_produts = torch.cross(
#             dst_node_edges, dst_equivariant_node_features, dim=-1
#         )

#         gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

#         dv = scaled_edge_dir + gated_features + gated_cross_products
#         ds = multiply_first_dim(scale_features, batch.invariant_node_features[src_node])

#         dv = scatter(dv, dst_node, dim=0)
#         ds = scatter(ds, dst_node, dim=0)

#         batch.equivariant_node_features += dv
#         batch.invariant_node_features += ds

#         return batch


# def multiply_first_dim(w, x):
#     with warnings.catch_warnings(record=True):
#         return (w.T * x.T).T


# class Update(torch.nn.Module):
#     def __init__(self, n_features=128):
#         super().__init__()
#         self.U = EquivariantLinear(n_features, n_features)
#         self.V = EquivariantLinear(n_features, n_features)
#         self.n_features = n_features
#         self.mlp = old_embeddings.MLP(2 * n_features, n_features, 3 * n_features)

#     def forward(self, batch):
#         v = batch.equivariant_node_features
#         s = batch.invariant_node_features

#         Vv = self.V(v)
#         Uv = self.U(v)

#         Vv_norm = Vv.norm(dim=-1)
#         Vv_squared_norm = Vv_norm**2

#         mlp_in = torch.cat([Vv_norm, s], dim=-1)

#         gates, scale_squared_norm, add_invariant_features = torch.split(
#             self.mlp(mlp_in), self.n_features, dim=-1
#         )

#         delta_v = multiply_first_dim(Uv, gates)
#         delta_s = Vv_squared_norm * scale_squared_norm + add_invariant_features

#         batch.invariant_node_features = batch.invariant_node_features + delta_s
#         batch.equivariant_node_features = batch.equivariant_node_features + delta_v

#         return batch


# class EquivariantLinear(torch.nn.Module):
#     def __init__(self, n_features_in, n_features_out):
#         super().__init__()
#         self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

#     def forward(self, x):
#         return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


# class Readout(torch.nn.Module):
#     def __init__(self, n_features=128, n_features_out=13):
#         super().__init__()
#         self.mlp = old_embeddings.MLP(n_features, n_features, 2 * n_features_out)
#         self.V = EquivariantLinear(n_features, n_features_out)
#         self.n_features_out = n_features_out

#     def forward(self, batch):
#         invariant_node_features_out, gates = torch.split(
#             self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
#         )

#         equivariant_node_features = self.V(batch.equivariant_node_features)
#         equivariant_node_features_out = multiply_first_dim(
#             equivariant_node_features, gates
#         )

#         batch.invariant_node_features = invariant_node_features_out
#         batch.equivariant_node_features = equivariant_node_features_out
#         return batch