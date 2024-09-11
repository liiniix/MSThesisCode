import torch
from Services.thesis_code_service import get_dataset, train_val_test_model_and_return_result, get_hop_to_nodesFeatureMean_for_proposed_model
from utility import make_code_reproducible, make_nvidia_faster_computation
from plot_helper import show_layerwise_max_accuracy
from tqdm import tqdm
from Services.make_dict_from_hop_json_service import make_json_node_hop_hopNodes_json

def get_accuracy_dependent_on_num_layers():
    combined_num_epoch = 200
    print(combined_num_epoch)
    DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

    dataset = get_dataset("acm")
    print("ok")

    layerwise_max_acc_for_proposed = []
    layerwise_max_acc_for_gcn = []
    layerwise_max_acc_for_graphsage_mean = []
    layerwise_max_acc_for_graphsage_max = []
    layerwise_max_acc_for_gat = []

    json_node_hop_hopNodes_cache = make_json_node_hop_hopNodes_json("acm", "MakeEdgelist/CppHelper")
    
    cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, json_node_hop_hopNodes_cache)

    for num_layers in tqdm(range(20, 30)):
        
        

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

def bong():
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

