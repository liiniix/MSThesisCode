import torch
from Services.thesis_code_service import get_dataset, train_val_test_model_and_return_result, get_hop_to_nodesFeatureMean_for_proposed_model, get_model_and_optimizer, train_val_test_model_and_return_result_for_lrgb
from utility import make_code_reproducible, make_nvidia_faster_computation
from plot_helper import show_layerwise_max_accuracy
from tqdm import tqdm
from Services.make_dict_from_hop_json_service import make_json_node_hop_hopNodes_json

def get_accuracy_dependent_on_num_layers():
    combined_num_epoch = 600
    print(combined_num_epoch)
    DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

    dataset = get_dataset("citeseer")
    print("ok")

    layerwise_max_acc_for_proposed = []
    layerwise_max_acc_for_gcn = []
    layerwise_max_acc_for_graphsage_mean = []
    layerwise_max_acc_for_graphsage_max = []
    layerwise_max_acc_for_gat = []

    json_node_hop_hopNodes_cache = make_json_node_hop_hopNodes_json("citeseer", "MakeEdgelist/CppHelper")
    
    cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset[0].to(DEVICE), 30, DEVICE, json_node_hop_hopNodes_cache)

    for num_layers in tqdm(range(0, 30)):
        
        

        proposed_output = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)
        layerwise_max_acc_for_proposed.append(
            max(proposed_output['test accuracy'])
        )

        gcn_output = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "gcn",
                                            combined_num_epoch)
        layerwise_max_acc_for_gcn.append(
            max(gcn_output['test accuracy'])
        )
        graphsage_output_mean = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "graphsage",
                                            combined_num_epoch,
                                            aggr="mean")
        
        layerwise_max_acc_for_graphsage_mean.append(
            max(graphsage_output_mean['test accuracy'])
        )
        
        
        
        graphsage_output_max = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "graphsage",
                                            combined_num_epoch,
                                            aggr="max")
        
        layerwise_max_acc_for_graphsage_max.append(
            max(graphsage_output_max['test accuracy'])
        )
        
        
        gat_output = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "gat",
                                            combined_num_epoch)
        layerwise_max_acc_for_gat.append(
            max(gat_output['test accuracy'])
        )
        

    
    return [layerwise_max_acc_for_proposed,
            layerwise_max_acc_for_gcn,
            layerwise_max_acc_for_graphsage_mean,
            layerwise_max_acc_for_graphsage_max,
            layerwise_max_acc_for_gat]

def make_lrgb_hop_level_feature_mean_cache():

    DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')
    for dataset_type in ['train', 'test', 'val']:

        dataset = get_dataset("lrgb", lrgb_split = dataset_type)

        for i in tqdm(range(len(dataset))):
            torch.cuda.empty_cache()
            json_node_hop_hopNodes_cache = make_json_node_hop_hopNodes_json(f"{i}", f"MakeEdgelist/CppHelper/LRGB_Pep/{dataset_type}")
            cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset[i].to(DEVICE), 30, DEVICE, json_node_hop_hopNodes_cache)

            torch.save(cached_acc_hop_level_featureMean, f"MakeEdgelist/CppHelper/LRGB_Pep/{dataset_type}/{i}.pth")

    return

    for i in range(len(dataset)):
        torch.cuda.empty_cache()
        json_node_hop_hopNodes_cache = make_json_node_hop_hopNodes_json(f"{i}", "MakeEdgelist/CppHelper/LRGB/train")
        cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset[i].to(DEVICE), 30, DEVICE, json_node_hop_hopNodes_cache)

        torch.save(cached_acc_hop_level_featureMean, f"MakeEdgelist/CppHelper/LRGB/train/{i}.pth")
        #cached_acc_hop_level_featureMean = torch.load(f"MakeEdgelist/CppHelper/LRGB/train/{i}.pth")
        #print(cached_acc_hop_level_featureMean)


def get_accuracy_dependent_on_num_layers_lrgb():

    DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

    train_datasets = get_dataset("lrgb", 'train')
    test_datasets = get_dataset("lrgb", 'test')
    val_datasets = get_dataset("lrgb", 'val')

    layerwise_max_acc_for_proposed = []
    layerwise_max_acc_for_gcn = []
    layerwise_max_acc_for_graphsage_mean = []
    layerwise_max_acc_for_graphsage_max = []
    layerwise_max_acc_for_gat = []

    num_layers = 30

    model, optimizer = get_model_and_optimizer(train_datasets,
                                               DEVICE,
                                               num_layers,
                                               "proposed")

    train_val_test_model_and_return_result_for_lrgb(train_datasets,
                                                    test_datasets,
                                                    val_datasets,
                                                    DEVICE,
                                                    model,
                                                    optimizer,
                                                    "proposed",
                                                    "MakeEdgelist/CppHelper/LRGB_Pep")

    model, optimizer = get_model_and_optimizer(train_datasets,
                                               DEVICE,
                                               num_layers,
                                               "gat")

    train_val_test_model_and_return_result_for_lrgb(train_datasets,
                                                    test_datasets,
                                                    val_datasets,
                                                    DEVICE,
                                                    model,
                                                    optimizer,
                                                    "gat",
                                                    "")

    model, optimizer = get_model_and_optimizer(train_datasets,
                                               DEVICE,
                                               num_layers,
                                               "gcn")

    train_val_test_model_and_return_result_for_lrgb(train_datasets,
                                                    test_datasets,
                                                    val_datasets,
                                                    DEVICE,
                                                    model,
                                                    optimizer,
                                                    "gcn",
                                                    "")
    
    model, optimizer = get_model_and_optimizer(train_datasets,
                                               DEVICE,
                                               num_layers,
                                               "graphsage")

    train_val_test_model_and_return_result_for_lrgb(train_datasets,
                                                    test_datasets,
                                                    val_datasets,
                                                    DEVICE,
                                                    model,
                                                    optimizer,
                                                    "graphsage",
                                                    "")

    
    
    
    
    return

    num_epoch = 2

    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        current_dataset_index = -1
        for data in datasets:
            data = data.to(DEVICE)

            current_dataset_index += 1
            cached_acc_hop_level_featureMean = torch.load(f"MakeEdgelist/CppHelper/LRGB/train/{current_dataset_index}.pth")

            num_layers = 2

            proposed_output = train_val_test_model_and_return_result(datasets,
                                        DEVICE,
                                        num_layers,
                                        "proposed",
                                        1,
                                        cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean,
                                        lrgb_index=current_dataset_index)
            #layerwise_max_acc_for_proposed.append(
            #    max(proposed_output['test accuracy'])
            #)

def bong():
    get_accuracy_dependent_on_num_layers_lrgb()
    return
    make_lrgb_hop_level_feature_mean_cache()
    return
    print("NEW")
    make_code_reproducible()
    make_nvidia_faster_computation()

    layerwise_max_acc = get_accuracy_dependent_on_num_layers()

    show_layerwise_max_accuracy(layerwise_max_acc)


#if __name__ == "__main__":
#    print("NEW")
#    make_code_reproducible()
#    make_nvidia_faster_computation()

#    layerwise_max_acc = get_accuracy_dependent_on_num_layers()

#    show_layerwise_max_accuracy(layerwise_max_acc)

