import json
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class HFDatasetWrapper(Dataset):
    def __init__(
        self, 
        hf_splited_dataset: HFDataset,
        transform: callable = None,
        ):
        self.transform = transform
        #
        self.hf_splited_dataset = hf_splited_dataset

    def __len__(self):
        return len(self.hf_splited_dataset)

    def __getitem__(self, idx):
        #
        res_dict = self.hf_splited_dataset[idx]
        #
        if self.transform:
            res_dict = self.transform(res_dict)
        #
        return res_dict

    def set_transform(self, transform_fn):
        self.transform = transform_fn