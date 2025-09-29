import os
from torch.utils.data import Dataset
import pandas as pd
from utils.audio_utils import load_audio, fix_audio_length
from ..custom_datasets import custom_dataset_register


class SpoofCelebDataset(Dataset):
    def __init__(
        self, 
        data_root: str = "/cache/datasets/spoofceleb/spoofceleb/",
        split: str = "eval",
        transform: callable = None,
        ):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        # 读取metadata
        meta_csv_path = os.path.join(data_root, f"metadata/{split}.csv")
        self.metadata = pd.read_csv(meta_csv_path)
        #
        self.metadata["audio_path"] = f"{data_root}/flac/{split}/" + self.metadata["file"]
        self.metadata["label"] = self.metadata["attack"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.loc[idx, "audio_path"]
        label = self.metadata.loc[idx, "label"]
        # 读取音频
        waveform, sample_rate = load_audio(audio_path)
        # 固定音频长度
        waveform, sample_rate = fix_audio_length(waveform, sample_rate, target_length_sec=4)
        #
        audio = {
            "array": waveform,
            "sampling_rate": sample_rate,
        }
        res_dict = {
            "audio": audio,
            "label": label,
        }
        #
        if self.transform:
            res_dict = self.transform(res_dict)
        #
        return res_dict

    def set_transform(self, transform_fn):
        self.transform = transform_fn


custom_dataset_register('SpoofCeleb', SpoofCelebDataset)
