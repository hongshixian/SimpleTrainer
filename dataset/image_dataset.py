from torch.utils.data import Dataset
from PIL import Image


class SimpleImageDataset(Dataset):
    """
    一个基础的SimpleImageDataset类, 继承torch dataset。
    返回pil格式的image和label，返回是dict格式。
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        初始化数据集
        :param image_paths: 图像路径列表
        :param labels: 标签列表
        :param transform: 可选的图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 索引
        :return: 包含图像和标签的字典
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 返回字典格式
        return {
            'image': image,
            'label': label
        }