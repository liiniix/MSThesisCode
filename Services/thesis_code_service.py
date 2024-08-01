import torch
from DatasetController.dataset_controller import get_cora_dataset, get_proposed_dataset, get_citeseer_dataset, get_pubmed_dataset, get_in_memeory_nell_dataset, get_lrgb_dataset
import torch.nn.functional as F
from Models.GraphSage.graph_sage_controller import get_graphsage_model
from Models.GCN.gcn_controller import get_gcn_model
from Models.GAT.gat_controller import get_gat_model
from Models.ProposedModel.proposed_model import get_proposed_model, get_hop_to_nodesFeatureMean
from datetime import datetime
from plot_helper import multilineplot, showProposedVsOther, compare_outputs
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_f1_score


torch.manual_seed(15)
is_wandb_initaited = False

DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

def __train(model,
            optimizer,
            data):
    
    #data = dataset[0].to(DEVICE)
    is_lrgb = True

    model.train()
    optimizer.zero_grad()
    out = model(data)
    if is_lrgb:
        loss = F.nll_loss(out, data.y)
    else:
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()

    loss_item = loss.item()
    return loss_item


def __test(model,
           optimizer,
           data):
    
    #data = dataset[0].to(DEVICE)


    model.eval()
    pred = model(data).argmax(dim=1)
    train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    train_acc = int(train_correct) / int(data.train_mask.sum())
    val_acc = int(val_correct) / int(data.val_mask.sum())
    test_acc = int(test_correct) / int(data.test_mask.sum())
    
    return train_acc, val_acc, test_acc

def train_and_show_stat(num_epoch,
                        model,
                        optimizer,
                        dataset,
                        prelude,
                        output,
                        output_legend_prelude,
                        lrgb_index=-1):
    
    is_lrgb = True

    if is_lrgb:
        loss_item = __train(model,
                            optimizer,
                            dataset[lrgb_index].to(DEVICE))
        print(loss_item)
        
        return
    
    for epoch in range(num_epoch):
        loss_item = __train(model,
                            optimizer,
                            dataset[0].to(DEVICE))
        train_acc, val_acc, test_acc = __test(model,
                                     optimizer,
                                     dataset[0].to(DEVICE))
        
        if epoch % 10 == 0:
            print(f"train accuracy: {train_acc} val_accuracy:{val_acc} test accuracy: {test_acc} loss: {loss_item}")
            
            output[f"train accuracy"].append(train_acc)
            output[f"test accuracy"].append(test_acc)
            output[f"loss"].append(loss_item)
            output['x'].append(epoch)

    

def get_dataset(dataset_name, lrgb_split='train'):

    if dataset_name=="cora":
        dataset = get_cora_dataset()

    elif dataset_name=="citeseer":
        dataset = get_citeseer_dataset()

    elif dataset_name == "pubmed":
        dataset = get_pubmed_dataset()

    elif dataset_name=="nell":
        dataset = get_in_memeory_nell_dataset()

    elif dataset_name=="lrgb":
        dataset = get_lrgb_dataset(split=lrgb_split)

    return dataset


def get_hop_to_nodesFeatureMean_for_proposed_model(data, max_k, DEVICE, json_node_hop_hopNodes=None):
    
    
    return get_hop_to_nodesFeatureMean(
        data,
        max_k,
        DEVICE,
        json_node_hop_hopNodes
    )

def get_model_and_optimizer(dataset,
              DEVICE,
              num_layers,
              model_name,
              **kwargs):
    
    if model_name == 'proposed':
        model = get_proposed_model(dataset,
                                   DEVICE,
                                   num_layers=num_layers,
                                   apply_attention=True,
                                   **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=5e-2)

    elif model_name == "gcn":
        model = get_gcn_model(dataset,
                          DEVICE,
                          num_layers=num_layers)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.016, weight_decay=5e-4)
        
    elif model_name == "graphsage":
        model = get_graphsage_model(dataset,
                        DEVICE,
                        num_layers=num_layers,
                                   **kwargs)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    elif model_name == "gat":
        model = get_gat_model(dataset,
                        DEVICE,
                        num_layers=num_layers)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    return model, optimizer

def train_val_test_model_and_return_result_for_lrgb(train_dataset,
                                                    test_dataset,
                                                    val_dataset,
                                                    DEVICE,
                                                    model,
                                                    optimizer,
                                                    model_name,
                                                    lrgb_dataset_path,
                                                    **kwargs):

    output = {
                f"train accuracy":       [],
                f"test accuracy":        [],
                f"loss":                [],
                'x':                   []
    }

    for epoch in range(150):

        for i, data in tqdm(enumerate(train_dataset)):
            if model_name == "proposed":
                cached_acc_hop_level_featureMean = torch.load(f"{lrgb_dataset_path}/train/{i}.pth")
                model.update_cache(cached_acc_hop_level_featureMean)

            model.train()
            optimizer.zero_grad()
            out = model(data.to(DEVICE))
            loss = F.nll_loss(out, torch.argmax(data.y, dim=1))
            loss.backward()
            optimizer.step()
            loss_item = loss.item()

        train_correct = 0
        train_preds = torch.empty((0, ), dtype=torch.int32)
        train_act = torch.empty((0, ), dtype=torch.int32)
        total_train_nodes = 0

        val_correct = 0
        total_val_nodes = 0
        val_preds = torch.empty((0, ), dtype=torch.int32)
        val_act = torch.empty((0, ), dtype=torch.int32)

        test_correct = 0
        total_test_nodes = 0
        test_preds = torch.empty((0, ), dtype=torch.int32)
        test_act = torch.empty((0, ), dtype=torch.int32)

        for i, data in tqdm(enumerate(train_dataset)):
            if model_name == "proposed":
                cached_acc_hop_level_featureMean = torch.load(f"{lrgb_dataset_path}/train/{i}.pth")
                model.update_cache(cached_acc_hop_level_featureMean)

            pred = model(data.to(DEVICE)).argmax(dim=1)
            train_preds = torch.cat((train_preds, pred.to('cpu')), 0)
            train_act = torch.cat((train_act, data.y.argmax(dim=1).to('cpu')), 0)

            train_correct += (pred == data.y).sum()

            total_train_nodes += data.num_nodes

        for i, data in tqdm(enumerate(val_dataset)):
            if model_name == "proposed":
                cached_acc_hop_level_featureMean = torch.load(f"{lrgb_dataset_path}/val/{i}.pth")
                model.update_cache(cached_acc_hop_level_featureMean)

            pred = model(data.to(DEVICE)).argmax(dim=1)
            val_preds = torch.cat((val_preds, pred.to('cpu')), 0)
            val_act = torch.cat((val_act, data.y.argmax(dim=1).to('cpu')), 0)

            val_correct += (pred == data.y).sum()

            total_val_nodes += data.num_nodes

        for i, data in tqdm(enumerate(test_dataset)):
            if model_name == "proposed":
                cached_acc_hop_level_featureMean = torch.load(f"{lrgb_dataset_path}/test/{i}.pth")
                model.update_cache(cached_acc_hop_level_featureMean)

            pred = model(data.to(DEVICE)).argmax(dim=1)
            test_preds = torch.cat((test_preds, pred.to('cpu')), 0)
            test_act = torch.cat((test_act, data.y.argmax(dim=1).to('cpu')), 0)

            test_correct += (pred == data.y).sum()

            total_test_nodes += data.num_nodes

        train_acc = train_correct / total_train_nodes
        val_acc = val_correct / total_val_nodes
        test_acc = test_correct / total_test_nodes

        train_f1 = multiclass_f1_score(train_preds, train_act, num_classes=val_dataset.num_classes, average="macro")

        
        print(f"train accuracy: {train_f1} val_accuracy:{val_acc} test accuracy: {test_acc}) loss: {loss_item}")
        

    


def train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           model_name,
                                           num_epoch,
                                           lrgb_index=-1,
                                           **kwargs):
    print(model_name)

    output = {
                f"train accuracy":       [],
                f"test accuracy":        [],
                f"loss":                [],
                'x':                   []
    }

    if model_name == 'proposed':
        model = get_proposed_model(dataset,
                                   DEVICE,
                                   num_layers=num_layers,
                                   apply_attention=True,
                                   **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=5e-2)
    elif model_name == "gcn":
        model = get_gcn_model(dataset,
                          DEVICE,
                          num_layers=num_layers)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.016, weight_decay=5e-4)
    elif model_name == "graphsage":
        model = get_graphsage_model(dataset,
                        DEVICE,
                        num_layers=num_layers,
                                   **kwargs)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    elif model_name == "gat":
        model = get_gat_model(dataset,
                        DEVICE,
                        num_layers=num_layers)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_and_show_stat(num_epoch,
                        model,
                        optimizer,
                        dataset,
                        model_name,
                        output,
                        model_name,
                        lrgb_index=lrgb_index)

    return output


    

def proposed_vs_other():
    num_layers = 10
    combined_num_epoch = 800

    dataset = get_dataset("cora")
    cached_acc_hop_level_featureMean = get_hop_to_nodesFeatureMean_for_proposed_model(dataset, num_layers, DEVICE)

    proposed_output = train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           "proposed",
                                           combined_num_epoch,
                                           cached_acc_hop_level_featureMean=cached_acc_hop_level_featureMean)

    gcn_output = train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           "gcn",
                                           combined_num_epoch)
    
    graphsage_output = train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           "graphsage",
                                           combined_num_epoch)
    
    
    gat_output = train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           "gat",
                                           combined_num_epoch)

    compare_outputs(["proposed", "gcn", "graphsage", "gat"],
                    [proposed_output, gcn_output, graphsage_output, gat_output])


if __name__ == "__main__":
    proposed_vs_other()
