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
    def __init__(self,
                 dataset,
                 num_layers=1):
        super(ProposedModel, self).__init__()

        if num_layers < 1:
            num_layers = 1
        self.flag = 0
        self.old = True

        self.attention =  torch.nn.Parameter(torch.randn((1, 4), # <- start with random weights (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                            requires_grad=True)

        self.num_layers = num_layers

        intermediate = int(dataset.num_features / 2)

        self.lin0 = torch.nn.Linear(dataset.num_features,
                                    intermediate,
                                    bias=False)#dataset.num_features/2)
        #torch.nn.init.kaiming_normal_(self.lin0.weight)
        self.act0 = torch.nn.Sigmoid()

        self.lin1 = torch.nn.Linear(dataset.num_features ,
                                    intermediate,
                                    bias=False)#dataset.num_features/2)
        #torch.nn.init.kaiming_normal_(self.lin1.weight)
        self.act1 = torch.nn.Sigmoid()

        self.lin2 = torch.nn.Linear(dataset.num_features ,
                                    intermediate,
                                    bias=False)#dataset.num_features/2)
        #torch.nn.init.kaiming_normal_(self.lin2.weight)
        self.act2 = torch.nn.Sigmoid()

        self.lin3 = torch.nn.Linear(dataset.num_features ,
                                    intermediate,
                                    bias=False)#dataset.num_features/2)
        #torch.nn.init.kaiming_normal_(self.lin3.weight)
        self.act3 = torch.nn.Sigmoid()

        self.final = torch.nn.Linear(intermediate,#dataset.num_features/2,
                                     dataset.num_classes)
        #torch.nn.init.kaiming_normal_(self.final.weight)
        self.act_final = torch.nn.Sigmoid()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.old:
            if self.flag == 0:
                self.acc_hop_zero_featureMean = []
                self.acc_hop_one_featureMean = []
                self.acc_hop_two_featureMean = []
                self.acc_hop_three_featureMean = []

                node_to_hop_to_nodesFeatureMean = get_node_to_hop_to_nodesFeatureMean(data, 3)

                for (node, hop_to_nodesFeatureMean) in node_to_hop_to_nodesFeatureMean.items():
                    self.acc_hop_zero_featureMean.append(hop_to_nodesFeatureMean[0])
                    self.acc_hop_one_featureMean.append(hop_to_nodesFeatureMean[1])
                    self.acc_hop_two_featureMean.append(hop_to_nodesFeatureMean[2])
                    self.acc_hop_three_featureMean.append(hop_to_nodesFeatureMean[3])

                self.acc_hop_zero_featureMean = torch.stack(self.acc_hop_zero_featureMean)
                self.acc_hop_one_featureMean = torch.stack(self.acc_hop_one_featureMean)
                self.acc_hop_two_featureMean = torch.stack(self.acc_hop_two_featureMean)
                self.acc_hop_three_featureMean = torch.stack(self.acc_hop_three_featureMean)

                self.flag = 1
            out0 = self.lin0(self.acc_hop_zero_featureMean)
            #print(out0)
            act0_ = self.act0(out0)

            out1 = self.lin1(self.acc_hop_one_featureMean)
            act1_ = self.act1(out1)

            out2 = self.lin2(self.acc_hop_two_featureMean)
            act2_ = self.act2(out2)

            out3 = self.lin3(self.acc_hop_three_featureMean)
            act3_ = self.act3(out3)

            normalized_attention = F.softmax(self.attention, dim=0)

            attended = act0_*normalized_attention[0,0] +\
                       act1_*normalized_attention[0,1] +\
                       act2_*normalized_attention[0,2] +\
                       act3_*normalized_attention[0,3]

            out = self.final(attended)

            return F.log_softmax(out, dim=1)
        
        if self.flag == 0:
            self.acc_hop_level_featureMean = {}
            for hop in range(self.num_layers + 1):
                self.acc_hop_level_featureMean[hop] = []

            node_to_hop_to_nodesFeatureMean = get_node_to_hop_to_nodesFeatureMean(data, 3)
            
            for (node, hop_to_nodesFeatureMean) in node_to_hop_to_nodesFeatureMean.items():
                for hop in range(self.num_layers+1):
                    self.acc_hop_level_featureMean[hop].append(hop_to_nodesFeatureMean[hop])

            for hop in range(self.num_layers+1):
                self.acc_hop_level_featureMean[hop] = torch.stack(self.acc_hop_level_featureMean[hop])
                
            self.flag = 1

        out0 = self.lin0(self.acc_hop_level_featureMean[0])
        act0 = self.act0(out0)

        out1 = self.lin1(self.acc_hop_level_featureMean[1])
        act1 = self.act1(out1)

        #out2 = self.lin2(self.acc_hop_level_featureMean[2])
        #act2 = self.act2(out2)

        #out3 = self.lin3(self.acc_hop_level_featureMean[3])
        #act3 = self.act_final(out3)

        

        out = self.final(self.act_final(act0+act1))#+act2+act3))
        return F.log_softmax(out, dim=1)


    

def get_node_to_hop_to_nodesFeatureMean(data, max_k):

    x = data.x
    G = to_networkx(data)
    node_to_hop_to_nodesFeatureMean = {}
    hop_to_nodesFeatureMean = {}
    
    for node in G.nodes:
        cc = nx.single_source_shortest_path_length(G, node, cutoff=max_k)
        for k in range(max_k+1):
            k_hop_nodes = [key for (key, value) in cc.items() if value == k]

            k_hop_nodes_features = torch.stack(
                                                    [torch.tensor(x[k_hop_node])
                                                    for k_hop_node in k_hop_nodes]
                                            ) if k_hop_nodes else torch.stack([torch.zeros(x[0].shape)])
            
            k_hop_nodesFeatureMean = torch.mean(k_hop_nodes_features, dim=0)

            hop_to_nodesFeatureMean[k] = k_hop_nodesFeatureMean
        
        node_to_hop_to_nodesFeatureMean[node] = hop_to_nodesFeatureMean
        hop_to_nodesFeatureMean = {}

    return node_to_hop_to_nodesFeatureMean


def get_proposed_model(dataset,
                        device,
                        num_layers):
    model = ProposedModel(dataset, num_layers)\
                    .to(device)
    return model