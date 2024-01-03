import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN


class GCNModel(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNModel, self).__init__()

        self.dataset = dataset
        
        self.gcn = GCN(dataset.num_features,
                       dataset.num_classes,
                       num_layers = 3)

    def forward(self):
        data = self.dataset[0]

        x = self.gcn(data.x,
                     data.edge_index)
        return F.log_softmax(x, dim=1)


def get_gcn_model(dataset,
                        device):
    model = GCNModel(dataset)\
                .to(device)
    return model