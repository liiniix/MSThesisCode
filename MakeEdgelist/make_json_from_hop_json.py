import json
import os.path
from tqdm import tqdm

def make_json_from_file(filename, input_folder):
    f = open(f"{input_folder}/{filename}.edgelist.txt") 

    hop_node_hopNodes = {}


    for line in tqdm(f):
        x = line.strip()\
                .split(':')
        
        node = int(x[0])
        hop = int(x[1])
        hopNodes = [int(hop_nodes_string) for hop_nodes_string in x[2].split(',')]
        
        if node not in hop_node_hopNodes:
            hop_node_hopNodes[node] = {}


        hop_node_hopNodes[node][hop] = hopNodes
    
    return hop_node_hopNodes

def is_jsonFile_available(filename, input_folder):
    return os.path.isfile(f"{input_folder}/{filename}.edgelist.txt")