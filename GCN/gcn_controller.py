import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN


class GCNModel(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNModel, self).__init__()
        
        self.gcn = GCN(dataset.num_features,
                       dataset.num_classes,
                       num_layers = 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gcn(x,
                     edge_index)
        return F.log_softmax(x, dim=1)


def get_gcn_model(dataset,
                        device):
    model = GCNModel(dataset)\
                .to(device)
    return model