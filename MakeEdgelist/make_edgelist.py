from Services.thesis_code_service import get_dataset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import subprocess

def make_edgelist(dataset_name, output_folder):
    dataset = get_dataset(dataset_name)

    G = to_networkx(dataset[0])
    nx.write_edgelist(G, f"{output_folder}/{dataset_name}.edgelist")

def make_jsonFile():
    subprocess.run(["g++", ".\MakeEdgelist\CppHelper\\abul.cpp", "-o", ".\MakeEdgelist\CppHelper\\abul"],
                   shell=True, check=True)
    
    subprocess.run(".\MakeEdgelist\CppHelper\\abul", shell=True, check=True)