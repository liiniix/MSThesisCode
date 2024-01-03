import torch
from DatasetController.dataset_controller import prepare_datasets, get_cora_dataset
import torch.nn.functional as F
from GraphSage.graph_sage_controller import get_graphsage_model
from GCN.gcn_controller import get_gcn_model


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
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def __test(model,
           optimizer,
           dataset):
    
    data = dataset[0].to(DEVICE)

    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train_and_show_stat(num_epoch,
                        model,
                        optimizer,
                        dataset,
                        prelude):
    print(prelude)
    best_val_acc = test_acc = 0

    for epoch in range(num_epoch):
        __train(model,
                optimizer,
                dataset)
        _, val_acc, tmp_test_acc = __test(model,
                                          optimizer,
                                          dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
        
        if epoch % 10 == 0:
            print(log.format(epoch, best_val_acc, test_acc))

prepare_datasets()

dataset = get_cora_dataset()
model = get_graphsage_model(dataset,
                            DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_and_show_stat(100,
                    model,
                    optimizer,
                    dataset,
                    "GraphSAGE")

model = get_gcn_model(dataset,
                      DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_and_show_stat(100,
                    model,
                    optimizer,
                    dataset,
                    "GCN")
