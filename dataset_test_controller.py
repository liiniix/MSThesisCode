from DatasetController.dataset_controller import get_cora_dataset, get_citeseer_dataset, get_pubmed_dataset, get_nell_dataset
import torch_geometric.transforms as T


if __name__ == "__main__":
    datasets = [
        ("Cora", get_cora_dataset()),
        ("CiteSeer", get_citeseer_dataset()),
        ("PubMed", get_pubmed_dataset())
    ]

    for dataset_name, dataset in datasets:
        
        print_string = f"{dataset_name} has {dataset[0].num_nodes} nodes"
        print_string += f"\t{dataset.num_classes} classes,"
        print_string += f"\t{dataset.num_features} features,"
        print_string += f"\t{len(dataset[0].x[dataset[0].train_mask])} train mask,"
        print_string += f"\t{len(dataset[0].x[dataset[0].val_mask])} val mask,"
        print_string += f"\t{len(dataset[0].x[dataset[0].test_mask])} test mask,"
        print_string += f"\t{len(dataset[0].x[dataset[0].train_mask]) + len(dataset[0].x[dataset[0].val_mask]) + len(dataset[0].x[dataset[0].test_mask])} total mask"
        
        print(print_string)
        