import torch
from DatasetController.dataset_controller import get_cora_dataset, get_proposed_dataset, get_citeseer_dataset, get_pubmed_dataset
import torch.nn.functional as F
from GraphSage.graph_sage_controller import get_graphsage_model
from GCN.gcn_controller import get_gcn_model
from ProposedModel.proposed_model import get_proposed_model
import wandb
from datetime import datetime


config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

run = wandb.init(project="config_example", config=config)


torch.manual_seed(15)

DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

run.name = f'check_wandb_{datetime.now()}_{DEVICE}'

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
                        prelude):
    print(prelude)

    for epoch in range(num_epoch):
        loss_item = __train(model,
                optimizer,
                dataset)
        train_acc, test_acc = __test(model,
                                     optimizer,
                                     dataset)
        
        if epoch % 10 == 0:
            print(f"Train accuracy: {train_acc} Test accuracy: {test_acc} Loss: {loss_item}")
            wandb.log({"train_acc": train_acc, "test_acc": test_acc, "loss": loss_item})

dataset = get_citeseer_dataset()
model = get_proposed_model(dataset,
                            DEVICE,
                            num_layers=3)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-2)

train_and_show_stat(200,
                    model,
                    optimizer,
                    dataset,
                    "Proposed")

model = get_gcn_model(dataset,
                      DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_and_show_stat(200,
                    model,
                    optimizer,
                    dataset,
                    "GCN")


model = get_graphsage_model(dataset,
                      DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_and_show_stat(200,
                    model,
                    optimizer,
                    dataset,
                    "GraphSage")
