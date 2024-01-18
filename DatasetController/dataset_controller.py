from torch_geometric.datasets import Planetoid, CitationFull
from Proposed.proposed_dataset import ProposedDataset

DATASET_ROOT_FOLDER = "Datasets"

def get_cora_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "CiteSeer")
    
    return dataset


def get_proposed_dataset():
    dataset = ProposedDataset(root = DATASET_ROOT_FOLDER)
    
    return dataset