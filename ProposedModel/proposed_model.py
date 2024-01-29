import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def get_block(in_f, out_f, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f, *args, **kwargs),
        torch.nn.Sigmoid()
    )

class ProposedModel(torch.nn.Module):
    def __init__(self,
                 dataset,
                 DEVICE,
                 num_layers=1,
                 apply_attention=False):
        super(ProposedModel, self).__init__()

        if num_layers < 1:
            num_layers = 1
        
        self.num_layers = num_layers
        self.apply_attention = apply_attention

        self.flag = 0

        self.DEVICE = DEVICE
        intermediate = int(dataset.num_features / 2)

        self.linears = torch.nn.ModuleList([get_block(dataset.num_features,
                                                      intermediate,
                                                      bias=False)
                                            for _ in range(self.num_layers+1)])

        if self.apply_attention:
            self.attention =  torch.nn.Parameter(torch.randn((1, self.num_layers+1),
                                                            dtype=torch.float
                                                            )
                                                                .to(self.DEVICE),
                                                requires_grad=True)

        self.final = torch.nn.Linear(intermediate,
                                     dataset.num_classes)
        self.act_final = torch.nn.Sigmoid()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.flag == 0:
            self.acc_hop_level_featureMean = {}
            for hop in range(self.num_layers + 1):
                self.acc_hop_level_featureMean[hop] = []

            node_to_hop_to_nodesFeatureMean = get_node_to_hop_to_nodesFeatureMean(data, 3, self.DEVICE)
            
            for (node, hop_to_nodesFeatureMean) in node_to_hop_to_nodesFeatureMean.items():
                for hop in range(self.num_layers+1):
                    self.acc_hop_level_featureMean[hop].append(hop_to_nodesFeatureMean[hop])

            for hop in range(self.num_layers+1):
                self.acc_hop_level_featureMean[hop] = torch.stack(self.acc_hop_level_featureMean[hop]).to(self.DEVICE)
                
            self.flag = 1

    
    
        linears_output = [self.linears[i](self.acc_hop_level_featureMean[i]) for i in range(self.num_layers+1)]
        
        if self.apply_attention:
            normalized_attention = F.softmax(self.attention, dim=0)

            attended = torch.zeros(linears_output[0].shape)\
                            .to(self.DEVICE)

            for i in range(self.num_layers+1):
                attended += linears_output[i] * normalized_attention[0,i]
        else:
            summed = torch.zeros(linears_output[0].shape)\
                            .to(self.DEVICE)

            for i in range(self.num_layers+1):
                summed += linears_output[i]
            
            summed = summed/(self.num_layers + 1)
        
        out = self.final(attended if self.apply_attention else summed)

        return F.log_softmax(out, dim=1)


def get_node_to_hop_to_nodesFeatureMean(data, max_k, DEVICE):

    x = data.x
    G = to_networkx(data)
    node_to_hop_to_nodesFeatureMean = {}
    hop_to_nodesFeatureMean = {}
    
    for node in G.nodes:
        cc = nx.single_source_shortest_path_length(G, node, cutoff=max_k)
        for k in range(max_k+1):
            k_hop_nodes = [key for (key, value) in cc.items() if value == k]

            k_hop_nodes_features = torch.stack(
                                                    [torch.tensor(x[k_hop_node]).to(DEVICE)
                                                    for k_hop_node in k_hop_nodes]
                                            ) if k_hop_nodes else torch.stack([torch.zeros(x[0].shape)]).to(DEVICE)
            
            k_hop_nodesFeatureMean = torch.mean(k_hop_nodes_features, dim=0).to(DEVICE)
            hop_to_nodesFeatureMean[k] = k_hop_nodesFeatureMean
        
        node_to_hop_to_nodesFeatureMean[node] = hop_to_nodesFeatureMean
        hop_to_nodesFeatureMean = {}

    return node_to_hop_to_nodesFeatureMean


def get_proposed_model(dataset,
                        device,
                        num_layers,
                        apply_attention=False):
    model = ProposedModel(dataset, device, num_layers, apply_attention=apply_attention)\
                    .to(device)
    return model