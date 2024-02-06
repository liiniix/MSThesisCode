import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT


class GATModel(torch.nn.Module):
    def __init__(self, dataset, num_layers, trial=None):
        super(GATModel, self).__init__()
        
        self.gat = GAT(dataset.num_features,
                       dataset.num_classes,
                       num_layers=num_layers,
                       aggr="max",
                       heads=dataset.num_classes*2,
                       concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_gat_model(dataset,
                        device,
                        num_layers,
                        optuna_trial=None):
    model = GATModel(dataset, num_layers, trial=optuna_trial)\
                    .to(device)
    return model


