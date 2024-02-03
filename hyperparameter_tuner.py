import optuna
from optuna.trial import TrialState

import torch
import torch.nn.functional as F

from GraphSage.graph_sage_controller import get_graphsage_model
from GCN.gcn_controller import get_gcn_model
from GAT.gat_controller import get_gat_model
from ProposedModel.proposed_model import get_proposed_model
from DatasetController.dataset_controller import get_cora_dataset, get_proposed_dataset, get_citeseer_dataset, get_pubmed_dataset, get_nell_dataset


DEVICE = torch.device('cuda'
                      if
                        torch.cuda.is_available()
                      else
                        'cpu')

cache_to_pass_between_trials = {}


def train(model,
          optimizer,
          dataset):
    
    data = dataset[0].to(DEVICE)

    model.train() 
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()




def test(model,
         optimizer,
         dataset):
    
    data = dataset[0].to(DEVICE)


    model.eval()
    pred = model(data).argmax(dim=1)

    val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    val_acc = int(val_correct) / int(data.val_mask.sum())
    
    return val_acc


def objective(trial):

    n_epochs = 700

    dataset = get_pubmed_dataset()


    num_layers = trial.suggest_int("num_layers", 0, 10)

    model = get_proposed_model(dataset,
                               DEVICE,
                               num_layers,
                               apply_attention=True, optuna_trial=trial,
                               cache_to_pass_between_trials=cache_to_pass_between_trials)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train(model, optimizer, dataset)  # Train the model
        accuracy = test(model, optimizer, dataset)   # Evaluate the model

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy



if __name__ == '__main__':
    torch.manual_seed(15)

    

    number_of_trials = 1000
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)


    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))