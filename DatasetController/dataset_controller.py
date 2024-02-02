from torch_geometric.datasets import Planetoid, CitationFull
from Proposed.proposed_dataset import ProposedDataset

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


def get_proposed_dataset():
    dataset = ProposedDataset(root = DATASET_ROOT_FOLDER)
    
    return dataset