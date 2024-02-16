from torch_geometric.datasets import Planetoid, CitationFull, NELL
from Proposed.proposed_dataset import ProposedDataset
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
import torch


DATASET_ROOT_FOLDER = "Datasets"
DATASET_ROOT_FOLDER_FOR_NELL = "Datasets/NELL"

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
    return dataset


def get_in_memeory_nell_dataset():
    dataset = InMemoryNellDataset(root = DATASET_ROOT_FOLDER_FOR_NELL)
    
    return random_split(dataset, is_nell=True)


def get_proposed_dataset():
    dataset = ProposedDataset(root = DATASET_ROOT_FOLDER)
    
    return dataset


def random_split(data, num_train_per_class: int = 20, num_val: int = 500, is_nell=False):
    data.train_mask.fill_(False)
    for c in range(data.num_classes):
        num_train_per_class = 2 if is_nell else num_train_per_class
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




import torch
from torch_geometric.data import InMemoryDataset, Data


class InMemoryNellDataset(InMemoryDataset):
    def __init__(self, root, data, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.load(self.processed_paths[0])
        self.dataset = get_nell_dataset()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['file.edges', 'file.x']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        
        dataset = get_nell_dataset()
        
        y  = dataset[0].y
        y_unique = y.unique()
        to_be_replaced_with = torch.arange(0, y_unique.shape[0])
        for i in range(y_unique.shape[0]):
            y[y==y_unique[i]] = to_be_replaced_with[i]
        
        
        data = Data(x=dataset[0].x.to_dense(),
                edge_index=dataset[0].edge_index,
                y=y)
        data.train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        
        
        data_list = [data]

        #self.save(data, self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])