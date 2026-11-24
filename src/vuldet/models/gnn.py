# Nguyễn Tâm Bình - 2324801030003
import torch
from torch import nn
from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

NUM_NODE_TYPES = 40
NUM_RELATIONS = 8


def with_reverse(edge_index, edge_type):

    return torch.cat([edge_index, edge_index.flip(0)], dim=1), torch.cat([edge_type, edge_type + 4])


def readout(h, batch):
    return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)


class GraphEncoder(nn.Module):
    def __init__(self, hidden=128, layers=3, llm_dim=0, dropout=0.2, num_node_types=NUM_NODE_TYPES):
        super().__init__()
        self.type_embedding = nn.Embedding(num_node_types, hidden)
        self.llm_projection = nn.Sequential(nn.Linear(llm_dim, hidden), nn.ReLU()) if llm_dim else None
        self.convs = nn.ModuleList(RGCNConv(hidden, hidden, NUM_RELATIONS) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_type, node_llm, edge_index, edge_type):
        h = self.type_embedding(node_type)
        if self.llm_projection is not None:
            h = h + self.llm_projection(node_llm)
        edge_index, edge_type = with_reverse(edge_index, edge_type)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(torch.relu(norm(conv(h, edge_index, edge_type))))
        return h


class GraphClassifier(nn.Module):
    def __init__(self, hidden=128, layers=3, llm_dim=0, func_dim=0, dropout=0.2, num_node_types=NUM_NODE_TYPES):
        super().__init__()
        self.encoder = GraphEncoder(hidden, layers, llm_dim, dropout, num_node_types)
        self.func_projection = nn.Sequential(nn.Linear(func_dim, hidden), nn.ReLU(), nn.Dropout(dropout)) if func_dim else None
        self.head = nn.Sequential(nn.Linear(2 * hidden + (hidden if func_dim else 0), hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, node_type, node_llm, edge_index, edge_type, batch, func=None):
        graph = readout(self.encoder(node_type, node_llm, edge_index, edge_type), batch)
        if self.func_projection is not None:
            graph = torch.cat([graph, self.func_projection(func)], dim=1)
        return self.head(graph).squeeze(-1)
