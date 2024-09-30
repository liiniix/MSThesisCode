import torch
import json
from Services.thesis_code_service import get_dataset, train_val_test_model_and_return_result, get_hop_to_nodesFeatureMean_for_proposed_model
from utility import make_code_reproducible, make_nvidia_faster_computation
from plot_helper import show_layerwise_max_accuracy
from tqdm import tqdm
from Services.make_dict_from_hop_json_service import make_json_node_hop_hopNodes_json

def get_accuracy_dependent_on_num_layers(dataset_name):
    combined_num_epoch = 600
    print(combined_num_epoch)
    DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

    dataset = get_dataset(dataset_name)
    print("ok")

    #layerwise_max_acc_for_proposed_mean = []
    #layerwise_max_acc_for_proposed_sum = []
    layerwise_max_acc_for_proposed_mean_5 = []
    layerwise_max_acc_for_proposed_sum_5 = []
    layerwise_max_acc_for_proposed_mean_10 = []
    layerwise_max_acc_for_proposed_sum_10 = []
    layerwise_max_acc_for_proposed_mean_15 = []
    layerwise_max_acc_for_proposed_sum_15 = []
    layerwise_max_acc_for_proposed_mean_20 = []
    layerwise_max_acc_for_proposed_sum_20 = []
    layerwise_max_acc_for_gcn = []
    layerwise_max_acc_for_graphsage_mean = []
    layerwise_max_acc_for_graphsage_sum = []
    layerwise_max_acc_for_graphsage_max = []
    layerwise_max_acc_for_gat = []

    json_node_hop_hopNodes_cache = make_json_node_hop_hopNodes_json(dataset_name, "MakeEdgelist/CppHelper")
    
    #cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, True, json_node_hop_hopNodes_cache)
    #cached_acc_hop_level_featureSum=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, False, json_node_hop_hopNodes_cache)

    for num_layers in tqdm(range(0, 31)):

        print(f"{dataset_name} {num_layers}")
        
        cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, True, 5, json_node_hop_hopNodes_cache)
        cached_acc_hop_level_featureSum=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, False, 5, json_node_hop_hopNodes_cache)

    
        proposed_output_mean = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)
        layerwise_max_acc_for_proposed_mean_5.append(
            max(proposed_output_mean['test accuracy'])
        )

        proposed_output_sum = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureSum)
        layerwise_max_acc_for_proposed_sum_5.append(
            max(proposed_output_sum['test accuracy'])
        )



        ##################



        cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, True, 10, json_node_hop_hopNodes_cache)
        cached_acc_hop_level_featureSum=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, False, 10, json_node_hop_hopNodes_cache)

    
        proposed_output_mean = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)
        layerwise_max_acc_for_proposed_mean_10.append(
            max(proposed_output_mean['test accuracy'])
        )

        proposed_output_sum = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureSum)
        layerwise_max_acc_for_proposed_sum_10.append(
            max(proposed_output_sum['test accuracy'])
        )



        #####################


        cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, True, 15, json_node_hop_hopNodes_cache)
        cached_acc_hop_level_featureSum=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, False, 15, json_node_hop_hopNodes_cache)

    
        proposed_output_mean = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)
        layerwise_max_acc_for_proposed_mean_15.append(
            max(proposed_output_mean['test accuracy'])
        )

        proposed_output_sum = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureSum)
        layerwise_max_acc_for_proposed_sum_15.append(
            max(proposed_output_sum['test accuracy'])
        )



        ###############################


        cached_acc_hop_level_featureMean=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, True, 20, json_node_hop_hopNodes_cache)
        cached_acc_hop_level_featureSum=get_hop_to_nodesFeatureMean_for_proposed_model(dataset, 30, DEVICE, False, 20, json_node_hop_hopNodes_cache)

    
        proposed_output_mean = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)
        layerwise_max_acc_for_proposed_mean_20.append(
            max(proposed_output_mean['test accuracy'])
        )

        proposed_output_sum = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "proposed",
                                            combined_num_epoch,
                                            cached_acc_hop_level_featureMean=cached_acc_hop_level_featureSum)
        layerwise_max_acc_for_proposed_sum_20.append(
            max(proposed_output_sum['test accuracy'])
        )




        #######################################

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
        
        
        graphsage_output_sum = train_val_test_model_and_return_result(dataset,
                                            DEVICE,
                                            num_layers,
                                            "graphsage",
                                            combined_num_epoch,
                                            aggr="sum")
        
        layerwise_max_acc_for_graphsage_sum.append(
            max(graphsage_output_sum['test accuracy'])
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
        

    
    return [layerwise_max_acc_for_proposed_mean_5,
            layerwise_max_acc_for_proposed_sum_5,
            layerwise_max_acc_for_proposed_mean_10,
            layerwise_max_acc_for_proposed_sum_10,
            layerwise_max_acc_for_proposed_mean_15,
            layerwise_max_acc_for_proposed_sum_15,
            layerwise_max_acc_for_proposed_mean_20,
            layerwise_max_acc_for_proposed_sum_20,
            layerwise_max_acc_for_gcn,
            layerwise_max_acc_for_graphsage_mean,
            layerwise_max_acc_for_graphsage_sum,
            layerwise_max_acc_for_graphsage_max,
            layerwise_max_acc_for_gat]

def bong():
    print("NEW")
    make_code_reproducible()
    make_nvidia_faster_computation()

    for dataset_name in ["citeseer", "cora", "pubmed"]:

        layerwise_max_acc = get_accuracy_dependent_on_num_layers(dataset_name)

        out_filename = f"{dataset_name}__epoch_600__0to30__PhD_rand.json"
        layerwise_max_acc_dict = {
            "HGAT-mean-5": layerwise_max_acc[0],
            "HGAT-sum-5": layerwise_max_acc[1],
            "HGAT-mean-10": layerwise_max_acc[2],
            "HGAT-sum-10": layerwise_max_acc[3],
            "HGAT-mean-15": layerwise_max_acc[4],
            "HGAT-sum-15": layerwise_max_acc[5],
            "HGAT-mean-20": layerwise_max_acc[6],
            "HGAT-sum-20": layerwise_max_acc[7],
            "GCN": layerwise_max_acc[8],
            "GraphSAGE-Mean": layerwise_max_acc[9],
            "GraphSAGE-Sum": layerwise_max_acc[10],
            "GraphSAGE-Max": layerwise_max_acc[11],
            "GAT": layerwise_max_acc[12]

        }

        layerwise_max_acc_dict_json = json.dumps(layerwise_max_acc_dict)

        with open(out_filename, 'w') as file:
            file.write(layerwise_max_acc_dict_json)

        show_layerwise_max_accuracy(layerwise_max_acc)


#if __name__ == "__main__":
#    print("NEW")
#    make_code_reproducible()
#    make_nvidia_faster_computation()

#    layerwise_max_acc = get_accuracy_dependent_on_num_layers()

#    show_layerwise_max_accuracy(layerwise_max_acc)

