import matplotlib.pyplot as plt
import scienceplots
import json
from scipy import interpolate
import numpy as np

colors = ["#ef233c", "#928e85", "#1c629d", "#748700", "#000000"]

isSavePGF = False

plt.style.use(['science', 'grid'])

def make_content(dataset_name):

    with open(f'{dataset_name}__epoch_200__0to30__PhD.json', 'r') as file:
        data = json.load(file)

    for i, (method_name, out) in enumerate(data.items()):

        x=np.arange(0, len(data[method_name]), 1)
        y = data[method_name]

        f = interpolate.PchipInterpolator(x, y)
        
        xnew = np.linspace(0, x[-1], num=3001)
        ynew = f(xnew)
        
        plt.plot(xnew, ynew, label=method_name, color=colors[i])



def make_final(dataset_name):
    textwidth = 4.7747
    aspect_ratio = 6/8
    scale = 1.0
    figwidth = textwidth * scale
    figheight = figwidth * aspect_ratio

    plt.style.use(["science", "grid"])
    import matplotlib
    if isSavePGF:
        matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.gca()
    make_content(dataset_name)
    legend = plt.legend(fancybox=False, edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))
    legend.get_frame().set_linewidth(0.5)
    width = 0.5
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)
    plt.xlabel("Number of GNN Layers, $K$")
    plt.ylabel("Node Classificaion Accuracy")
    if isSavePGF:
        plt.savefig(f"{dataset_name}__epoch_200__0to30__PhD.pgf")
    else:
        plt.show()

for dataset_name in ['cora', 'citeseer', 'pubmed']:
    make_final(dataset_name)
