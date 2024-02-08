import torch
from DatasetController.dataset_controller import get_cora_dataset, get_proposed_dataset, get_citeseer_dataset, get_pubmed_dataset, get_nell_dataset
import torch.nn.functional as F
from GraphSage.graph_sage_controller import get_graphsage_model
from GCN.gcn_controller import get_gcn_model
from GAT.gat_controller import get_gat_model
from ProposedModel.proposed_model import get_proposed_model
from datetime import datetime
from plot_helper import multilineplot, showProposedVsOther, compare_outputs


torch.manual_seed(15)
is_wandb_initaited = False

DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')


def __train(model,
            optimizer,
            dataset):
    
    data = dataset[0].to(DEVICE)

    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()

    loss_item = loss.item()
    return loss_item


def __test(model,
           optimizer,
           dataset):
    
    data = dataset[0].to(DEVICE)


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
                        output_legend_prelude):
    
    for epoch in range(num_epoch):
        loss_item = __train(model,
                            optimizer,
                            dataset)
        train_acc, val_acc, test_acc = __test(model,
                                     optimizer,
                                     dataset)
        
        if epoch % 10 == 0:
            print(f"train accuracy: {train_acc} val_accuracy:{val_acc} test accuracy: {test_acc} loss: {loss_item}")
            
            output[f"train accuracy"].append(train_acc)
            output[f"test accuracy"].append(test_acc)
            output[f"loss"].append(loss_item)
            output['x'].append(epoch)

def get_dataset(dataset_name):

    if dataset_name=="cora":
        dataset = get_cora_dataset()

    elif dataset_name=="citeseer":
        dataset = get_citeseer_dataset()

    elif dataset_name == "pubmed":
        dataset = get_pubmed_dataset()

    elif dataset_name=="nell":
        dataset = get_nell_dataset()

    return dataset


def train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           model_name,
                                           num_epoch):
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
                                   apply_attention=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=5e-2)
    elif model_name == "gcn":
        model = get_gcn_model(dataset,
                          DEVICE,
                          num_layers=num_layers)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.016, weight_decay=5e-4)
    elif model_name == "graphsage":
        model = get_graphsage_model(dataset,
                        DEVICE,
                        num_layers=num_layers)
    
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
                        model_name)

    return output


    

def proposed_vs_other():
    num_layers = 20
    combined_num_epoch = 800

    dataset = get_dataset("cora")

    proposed_output = train_val_test_model_and_return_result(dataset,
                                           DEVICE,
                                           num_layers,
                                           "proposed",
                                           combined_num_epoch)

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


proposed_vs_other()

