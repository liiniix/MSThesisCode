from torch_geometric.datasets import Planetoid

DATASET_ROOT_FOLDER = "Datasets"

def prepare_datasets():
    Planetoid(root = DATASET_ROOT_FOLDER,
              name= "Cora")

def get_cora_dataset():
    dataset = Planetoid(root = DATASET_ROOT_FOLDER,
                        name= "Cora")
    
    return dataset