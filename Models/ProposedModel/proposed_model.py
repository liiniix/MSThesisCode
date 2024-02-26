import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                 apply_attention=False,
                 trial=None,
                 cached_acc_hop_level_featureMean=None):
        super(ProposedModel, self).__init__()

        if num_layers < 0:
            num_layers = 1
        
        self.num_layers = num_layers
        self.apply_attention = apply_attention

        self.DEVICE = DEVICE
        self.acc_hop_level_featureMean = cached_acc_hop_level_featureMean
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


def get_node_to_hop_to_nodesFeatureMean(data, max_k, DEVICE, json_node_hop_hopNodes=None):

    x = data.x.to(DEVICE)
    G = to_networkx(data)
    node_to_hop_to_nodesFeatureMean = {}
    hop_to_nodesFeatureMean = {}
    
    for node in tqdm(G.nodes):
        if json_node_hop_hopNodes is None:
            cc = nx.single_source_shortest_path_length(G, node, cutoff=max_k)
        for k in range(max_k+1):
            if json_node_hop_hopNodes:
                try:
                    k_hop_nodes = json_node_hop_hopNodes[str(node)][str(k)]
                except KeyError:
                    k_hop_nodes = []
                k_hop_nodes = [int(k_hop_node) for k_hop_node in k_hop_nodes]
            else:
                k_hop_nodes = [key for (key, value) in cc.items() if value == k]

            k_hop_nodesFeatureMean = torch.stack([torch.zeros(x[0].shape)]).to(DEVICE)

            if k_hop_nodes:

                k_hop_nodes_index = torch.tensor(k_hop_nodes).to(DEVICE)
                k_hop_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=DEVICE)\
                                    .scatter_(0, k_hop_nodes_index, True)
                
            k_hop_nodesFeatureMean = torch.mean(x[k_hop_mask], dim=0)            

            hop_to_nodesFeatureMean[k] = k_hop_nodesFeatureMean

        
        node_to_hop_to_nodesFeatureMean[node] = hop_to_nodesFeatureMean
        hop_to_nodesFeatureMean = {}

    return node_to_hop_to_nodesFeatureMean

def get_hop_to_nodesFeatureMean(data, max_k, DEVICE, json_node_hop_hopNodes=None):
    acc_hop_level_featureMean = {}
    for hop in range(max_k + 1):
        acc_hop_level_featureMean[hop] = []


    node_to_hop_to_nodesFeatureMean = get_node_to_hop_to_nodesFeatureMean(data, max_k, DEVICE, json_node_hop_hopNodes)
    
    for (node, hop_to_nodesFeatureMean) in node_to_hop_to_nodesFeatureMean.items():
        for hop in range(max_k+1):
            acc_hop_level_featureMean[hop].append(hop_to_nodesFeatureMean[hop])

    for hop in range(max_k+1):
        acc_hop_level_featureMean[hop] = torch.stack(acc_hop_level_featureMean[hop]).to(DEVICE)
        
    return acc_hop_level_featureMean


def get_proposed_model(dataset,
                        device,
                        num_layers,
                        apply_attention=False,
                        optuna_trial=None,
                        **kwargs):
    model = ProposedModel(dataset, device, num_layers, apply_attention=apply_attention, trial=optuna_trial, **kwargs)\
                    .to(device)
    return model