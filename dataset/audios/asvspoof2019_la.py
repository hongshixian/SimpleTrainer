from torch.utils.data import Dataset
import pandas as pd
from utils.audio_utils import load_audio, fix_audio_length
from ..custom_datasets import custom_dataset_register


class ASVspoof2019_LA_Dataset(Dataset):
    def __init__(
        self, 
        data_root: str = "F:/datasets/ASVSpoof2019LA/",
        split: str = "train",
        transform: callable = None,
        ):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        # 读取metadata
        suffix_map = {"train": "trn", "dev": "trl", "eval": "trl"}
        self.metadata = pd.read_csv(
            f"{data_root}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{suffix_map[split]}.txt",
             sep=" ", header=None)
        #
        if split == "dev":
            self.metadata = self.metadata.sample(frac=0.1, random_state=42).reset_index(drop=True)
        #
        self.metadata.columns = ["audio_name", "file_name", "_", "attack", "is_spoof"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_name = self.metadata.loc[idx, "file_name"]
        label = self.metadata.loc[idx, "is_spoof"]
        # 读取音频
        audio_path = f"{self.data_root}/ASVspoof2019_LA_{self.split}/flac/{audio_name}.flac"
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


custom_dataset_register('ASVspoof2019LA', ASVspoof2019_LA_Dataset)
