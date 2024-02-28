import json
import os.path

def make_json_from_file(filename, input_folder):
    json_file = open(f"{input_folder}/{filename}.edgelist.json") 
    loaded_json_data = json.load(json_file)
    return loaded_json_data

def is_jsonFile_available(filename, input_folder):
    return os.path.isfile(f"{input_folder}/{filename}.edgelist.json")