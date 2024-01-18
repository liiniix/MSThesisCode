import torch
from torch_geometric.data import InMemoryDataset, Data


class ProposedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['file.edges', 'file.x']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        x = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float)
        y = torch.tensor([2, 2, 2, 2, 2])


        data = Data(x=x, edge_index=edge_index, y=y)

        data.train_mask = torch.tensor([True, True, True, True, True],
                                  dtype=torch.bool)
        
        data.test_mask = torch.tensor([True, True, True, True, True],
                                  dtype=torch.bool)
        
        data.val_mask = torch.tensor([True, True, True, True, True],
                                  dtype=torch.bool)
        data_list = [data]

        #self.save(data, self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])