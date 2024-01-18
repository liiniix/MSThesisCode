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

class ProposedModel(torch.nn.Module):
    def __init__(self, dataset,
                 num_layers=1):
        super(ProposedModel, self).__init__()

        if num_layers < 1:
            num_layers = 1
        self.flag = 0

        self.num_layers = num_layers

        self.transform_neighbour = torch.nn.Linear(dataset.num_features ,
                                                   dataset.num_classes)
        self.act1 = torch.nn.Sigmoid()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        G = to_networkx(data)

        if self.flag == 0:
            self.flag = 1
            node_to_hop_to_nodesFeatureMean = get_hop_to_nodesFeatureMean(data, 2)
            pretty(node_to_hop_to_nodesFeatureMean)

        acc_self_and_neigh = []
        for node in G.nodes:
            cc = nx.single_source_shortest_path_length(G, node, cutoff=1)

            hop_one_nodes = [k for (k,v) in cc.items() if v == 1]

            hop_one_nodes_features = torch.stack(
                                                    [torch.tensor(x[hop_one_node])
                                                    for hop_one_node in hop_one_nodes]
                                                )
            bogor_tensor = x[node]
            
            #current_node_s_hop_to_feature_mean = get_hop_to_nodesFeatureMean(node, data, self.num_layers)

            for hope_one_node_feature in hop_one_nodes_features:
                bogor_tensor.add(hope_one_node_feature)
            
            acc_self_and_neigh.append(bogor_tensor)

        acc_self_and_neigh = torch.stack(acc_self_and_neigh)
        out = self.transform_neighbour(acc_self_and_neigh)
        out = self.act1(out)
        return F.log_softmax(out, dim=1)
    

def get_hop_to_nodesFeatureMean(data, max_k):

    x = data.x
    G = to_networkx(data)
    node_to_hop_to_nodesFeatureMean = {}
    hop_to_nodesFeatureMean = {}
    
    for node in G.nodes:
        cc = nx.single_source_shortest_path_length(G, node, cutoff=max_k)
        for k in range(1, max_k+1):
            k_hop_nodes = [key for (key, value) in cc.items() if value == k]

            k_hop_nodes_features = torch.stack(
                                                    [torch.tensor(x[k_hop_node])
                                                    for k_hop_node in k_hop_nodes]
                                            ) if k_hop_nodes else torch.tensor([])
            
            k_hop_nodesFeatureMean = torch.mean(k_hop_nodes_features, dim=0)

            hop_to_nodesFeatureMean[k] = k_hop_nodesFeatureMean
        
        node_to_hop_to_nodesFeatureMean[node] = hop_to_nodesFeatureMean

    return node_to_hop_to_nodesFeatureMean


def get_proposed_model(dataset,
                        device):
    model = ProposedModel(dataset)\
                    .to(device)
    return model