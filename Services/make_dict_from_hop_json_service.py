from MakeEdgelist.make_json_from_hop_json import make_json_from_file, is_jsonFile_available
from MakeEdgelist.make_edgelist import make_edgelist, make_jsonFile


def make_json_node_hop_hopNodes_json(file, path):

    is_json_available_ = is_jsonFile_available(file, path)

    if not is_json_available_:
        make_edgelist(file, path)
        make_jsonFile()

    return make_json_from_file(file, path)
