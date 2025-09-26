CUSTOM_DATASET_DICT = {}


def custom_dataset_register(name: str, dataset_cls):
    """
    注册自定义数据集类
    :param name: 数据集名称
    :param dataset_cls: 数据集类
    """
    global CUSTOM_DATASET_DICT
    CUSTOM_DATASET_DICT[name] = dataset_cls