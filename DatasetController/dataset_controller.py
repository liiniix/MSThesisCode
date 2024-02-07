from torch_geometric.datasets import Planetoid, CitationFull, NELL
from Proposed.proposed_dataset import ProposedDataset
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
import torch


DATASET_ROOT_FOLDER = "Datasets"

def get_citeseer_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "CiteSeer",
                        split='random')
    
    return random_split(dataset)

def get_cora_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "Cora",
                        split='random')
    
    return random_split(dataset)


def get_pubmed_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "PubMed",
                        split='random')
    
    return random_split(dataset)

def get_nell_dataset():

    transform = T.RandomNodeSplit(split='random')

    dataset = NELL(root = DATASET_ROOT_FOLDER, transform=transform)
    
    #return dataset

    num_train_per_class = 20
    num_val=500
    num_test=1000




    data = Data(x=dataset[0].x.to_dense(),
                edge_index=dataset[0].edge_index,
                y=dataset[0].y,
                train_mask=dataset[0].train_mask,
                test_mask=dataset[0].test_mask,
                val_mask=dataset[0].val_mask)
    data.train_mask.fill_(False)
    for c in range(dataset.num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        data.val_mask.fill_(False)
        data.val_mask[remaining[:num_val]] = True

        data.test_mask.fill_(False)
        data.test_mask[remaining[num_val:num_val + num_test]] = True


    return data

def get_proposed_dataset():
    dataset = ProposedDataset(root = DATASET_ROOT_FOLDER)
    
    return dataset

def random_split(data, num_train_per_class: int = 20, num_val: int = 500):
    data.train_mask.fill_(False)
    for c in range(data.num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    data.test_mask[remaining[num_val:]] = True

    return data