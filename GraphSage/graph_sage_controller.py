import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, dataset):
        super(GraphSAGEModel, self).__init__()

        self.dataset = dataset
        
        self.graph_sage = GraphSAGE(dataset.num_features,
                             dataset.num_classes,
                             num_layers = 3,
                             aggr="max")

    def forward(self):
        data = self.dataset[0]

        x = self.graph_sage(data.x,
                            data.edge_index)
        return F.log_softmax(x, dim=1)


def get_graphsage_model(dataset,
                        device):
    model = GraphSAGEModel(dataset)\
                    .to(device)
    return model