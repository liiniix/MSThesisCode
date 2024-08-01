from torch_geometric.datasets import Planetoid, CitationFull, NELL, LRGBDataset
from Proposed.proposed_dataset import ProposedDataset
import torch_geometric.transforms as T


DATASET_ROOT_FOLDER = "Datasets"

def get_citeseer_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "CiteSeer",
                        split='random')
    
    return dataset

def get_cora_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "Cora",
                        split='random')
    
    return dataset


def get_pubmed_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "PubMed",
                        split='random')
    
    return dataset

def get_nell_dataset():

    transform = T.RandomNodeSplit(split='random')

    dataset = NELL(root = DATASET_ROOT_FOLDER, transform=transform)
    
    return dataset


def get_proposed_dataset():
    dataset = ProposedDataset(root = DATASET_ROOT_FOLDER)
    
    return dataset

def get_lrgb_dataset(split='train'):
    dataset = LRGBDataset(root = DATASET_ROOT_FOLDER,
                        name= "Peptides-func",
                        split=split)
    
    return dataset


lrgb_train = get_lrgb_dataset("train")
lrgb_test = get_lrgb_dataset("test")
lrgb_val = get_lrgb_dataset("val")


from torch_geometric.utils.convert import to_networkx
import networkx as nx
print("kicu")
G = to_networkx(lrgb_train[0], edge_attrs =  {'a': lrgb_train[0].edge_attr})