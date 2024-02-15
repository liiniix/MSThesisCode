import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, trial=None, **kwargs):
        super(GraphSAGEModel, self).__init__()
        
        self.graph_sage = GraphSAGE(dataset.num_features,
                                    dataset.num_classes,
                                    num_layers=num_layers,
                                    trial=trial,
                                    **kwargs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.graph_sage(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_graphsage_model(dataset,
                        device,
                        num_layers,
                        optuna_trial=None,
                        **kwargs):
    model = GraphSAGEModel(dataset, num_layers=num_layers, trial=optuna_trial, **kwargs)\
                    .to(device)
    return model


