import matplotlib.pyplot as plt
import wandb
from datetime import datetime

def multilineplot(data, filename):
    '''
    data:
        data1_y: [],
        data2_y: [],
        ...
        x:       []
    '''
    for (key, value) in data.items():
        if key == 'x':
            continue

        plt.plot(data['x'], value, label=key)

    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.clf()


def showProposedVsOther(data1, data2, data3):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    for (key, value) in data1.items():
        if key == 'x':
            continue

        ax1.plot(data1['x'], value, label=key)

    
    for (key, value) in data2.items():
        if key == 'x':
            continue

        ax2.plot(data2['x'], value, label=key)


    for (key, value) in data3.items():
        if key == 'x':
            continue

        ax3.plot(data3['x'], value, label=key)

    plt.legend()
    plt.savefig("showProposedVsOther.png")
    plt.clf()


def initiate_wandb():
    
    config = {
        "hidden_layer_sizes": [32, 64],
        "kernel_sizes": [3],
        "activation": "ReLU",
        "pool_sizes": [2],
        "dropout": 0.5,
        "num_classes": 10,
    }

    run = wandb.init(project="MSThesisCode", config=config)

    run.name = f'msthesiscode_{datetime.now()}'


def compare_outputs(preludes, data):
    initiate_wandb()

    ys_train_acc = [dat['train accuracy'] for dat in data]
    ys_test_acc = [dat['test accuracy'] for dat in data]
    ys_loss = [dat['loss'] for dat in data]
    xs = data[0]['x']

    wandb.log({"train acc" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys_train_acc,
                       keys=preludes,
                       title="Train Acc",
                       xname="x units")})

    
    wandb.log({"test acc" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys_test_acc,
                       keys=preludes,
                       title="Test Acc",
                       xname="x units")})
    
    wandb.log({"loss" : wandb.plot.line_series(
                       xs=xs,
                       ys=ys_loss,
                       keys=preludes,
                       title="Loss",
                       xname="x units")})
    


def show_layerwise_max_accuracy(layerwise_max_accuracy):
    initiate_wandb()



    xs = range(len(layerwise_max_accuracy))

    wandb.log({"acc" : wandb.plot.line_series(
                       xs=xs,
                       ys=layerwise_max_accuracy,
                       keys=["Proposed", "GCN", "GraphSage", "GAT"],
                       title="Test Acc",
                       xname="x units")})

