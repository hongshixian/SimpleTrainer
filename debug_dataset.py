from dataset import get_dataset


dataset_args = {
    "custom_dataset_name": "ASVSpoof2019LA",
    "train_split": "train",
    "eval_split": "dev",
}


dataset = get_dataset(dataset_args)
print(dataset["train_dataset"][0])

