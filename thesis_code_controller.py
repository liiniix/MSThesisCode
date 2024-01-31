import torch
from DatasetController.dataset_controller import get_cora_dataset, get_proposed_dataset, get_citeseer_dataset, get_pubmed_dataset
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
    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    train_acc = int(train_correct) / int(data.train_mask.sum())
    test_acc = int(test_correct) / int(data.test_mask.sum())
    
    return train_acc, test_acc

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
        train_acc, test_acc = __test(model,
                                     optimizer,
                                     dataset)
        
        if epoch % 10 == 0:
            print(f"train accuracy: {train_acc} test accuracy: {test_acc} loss: {loss_item}")
            
            output[f"train accuracy"].append(train_acc)
            output[f"test accuracy"].append(test_acc)
            output[f"loss"].append(loss_item)
            output['x'].append(epoch)

dataset = get_citeseer_dataset()



def attention_vs_not():
    output_legend_prelude = "not attention"
    not_attention_output = {f"{output_legend_prelude} train accuray":       [],
                            f"{output_legend_prelude} test accuray":        [],
                            f"{output_legend_prelude} loss":                [],
                            'x':                                            []}

    model = get_proposed_model(dataset,
                               DEVICE,
                               num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-2)
    train_and_show_stat(500,
                        model,
                        optimizer,
                        dataset,
                        "Proposed",
                        not_attention_output,
                        output_legend_prelude)
    
    print(not_attention_output)
    multilineplot(not_attention_output, "not_attention")
    

    output_legend_prelude = "attention"
    attention_output = {f"{output_legend_prelude} train accuray":       [],
                        f"{output_legend_prelude} test accuray":        [],
                        f"{output_legend_prelude} loss":                [],
                        'x':                                            []}
    
    model = get_proposed_model(dataset,
                               DEVICE,
                               num_layers=3,
                               apply_attention=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-2)
    train_and_show_stat(500,
                        model,
                        optimizer,
                        dataset,
                        "Proposed",
                        attention_output,
                        output_legend_prelude)

    multilineplot(attention_output, "attention")


    

def proposed_vs_other():
    combined_num_epoch = 700
    output_legend_prelude = "proposed"
    proposed_output = {f"train accuracy":       [],
                       f"test accuracy":        [],
                       f"loss":                [],
                       'x':                                            []}
    model = get_proposed_model(dataset,
                               DEVICE,
                               num_layers=3,
                               apply_attention=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-2)
    train_and_show_stat(combined_num_epoch,
                        model,
                        optimizer,
                        dataset,
                        "Proposed",
                        proposed_output,
                        output_legend_prelude)







    output_legend_prelude = "gcn"
    gcn_output = {f"train accuracy":       [],
                        f"test accuracy":        [],
                        f"loss":                [],
                        'x':                                            []}
    
    model = get_gcn_model(dataset,
                          DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_and_show_stat(combined_num_epoch,
                        model,
                        optimizer,
                        dataset,
                        "GCN",
                        gcn_output,
                        output_legend_prelude)
    





    output_legend_prelude = "graphsage"
    graphsage_output = {f"train accuracy":       [],
                        f"test accuracy":        [],
                        f"loss":                [],
                        'x':                                            []}
    
    model = get_graphsage_model(dataset,
                        DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_and_show_stat(combined_num_epoch,
                      model,
                      optimizer,
                      dataset,
                      "GraphSage",
                        graphsage_output,
                        output_legend_prelude)
    









    
    output_legend_prelude = "gat"
    gat_output = {f"train accuracy":       [],
                        f"test accuracy":        [],
                        f"loss":                [],
                        'x':                                            []}
    
    model = get_gat_model(dataset,
                        DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_and_show_stat(combined_num_epoch,
                      model,
                      optimizer,
                      dataset,
                      "GAT",
                        gat_output,
                        output_legend_prelude)

    #showProposedVsOther(proposed_output, gcn_output, graphsage_output)
    compare_outputs(["proposed", "gcn", "graphsage", "gat"],
                    [proposed_output, gcn_output, graphsage_output, gat_output])



#attention_vs_not()

proposed_vs_other()