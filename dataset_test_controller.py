from DatasetController.dataset_controller import get_cora_dataset, get_citeseer_dataset, get_pubmed_dataset, get_nell_dataset
import torch_geometric.transforms as T


if __name__ == "__main__":
    datasets = [
        ("Cora", get_cora_dataset()),
        ("CiteSeer", get_citeseer_dataset()),
        ("PubMed", get_pubmed_dataset()),
        ("NELL", get_nell_dataset())
    ]

    for dataset_name, dataset in datasets:
        
        print(f"{dataset_name} has {dataset.len()} objects,\
                                   {dataset[0].num_nodes} nodes,\
                                   {dataset.num_classes} nodes,\
                                   {dataset.num_features} features\
                                   {len(dataset[0].x[dataset[0].train_mask])} train mask,\
                                   {len(dataset[0].x[dataset[0].val_mask])} val mask,\
                                   {len(dataset[0].x[dataset[0].test_mask])} test mask,")
        
        