import numpy as np
import torch
import torchaudio


def load_audio(audio_path: str, sample_rate: int = 16000, normalize: bool = True) -> torch.Tensor:
    """
    加载音频文件并转换为张量
    
    Args:
        audio_path: 音频文件路径
        sample_rate: 目标采样率，默认为16000
        normalize: 是否归一化音频，默认为True
    
    Returns:
        waveform: 音频波形张量，形状为 (channels, samples) 或 (samples,)
    """
    # 使用torchaudio加载音频
    waveform, sr = torchaudio.load(audio_path, normalize=normalize)
    
    # 如果采样率不同，进行重采样
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    return waveform, sample_rate

def save_audio(waveform: torch.Tensor, audio_path: str, sample_rate: int = 16000):
    """
    保存音频张量到文件
    
    Args:
        waveform: 音频波形张量，形状为 (channels, samples) 或 (samples,)
        audio_path: 输出音频文件路径
        sample_rate: 音频采样率，默认为16000
    """
    # 如果是单通道，添加通道维度
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # 确保waveform是float32类型
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    
    # 保存音频
    torchaudio.save(audio_path, waveform, sample_rate)

def fix_audio_length(waveform, sample_rate, target_length_sec=4, mode='start', pad_mode='right'):
    """
    将音频波形调整到指定长度
    
    Args:
        waveform: 输入的音频波形张量，形状为 (channels, samples) 或 (samples,)
        sample_rate: 音频采样率
        target_length_sec: 目标长度（秒），默认为4秒
        mode: 截断模式 ('start', 'center', 'end', 'random')
        pad_mode: 填充模式 ('left', 'right', 'center')
    
    Returns:
        waveform: 调整后的音频波形张量
        sample_rate: 采样率（保持不变）
    """
    # 确保输入是张量
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    
    # 计算目标长度（样本数）
    target_length = int(target_length_sec * sample_rate)
    
    # 获取当前音频长度和通道数
    current_length = waveform.size(-1)
    num_channels = waveform.size(0) if waveform.dim() > 1 else 1
    
    # 如果是单通道，确保是2维张量
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # 调整音频长度
    if current_length < target_length:
        # 如果音频太短，进行padding
        padding_size = target_length - current_length
        
        if pad_mode == 'right':
            # 右对齐：在末尾进行padding（默认）
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))
        elif pad_mode == 'left':
            # 左对齐：在开头进行padding
            waveform = torch.nn.functional.pad(waveform, (padding_size, 0))
        elif pad_mode == 'center':
            # 居中对齐：在两端均匀padding
            left_pad = padding_size // 2
            right_pad = padding_size - left_pad
            waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
        else:
            raise ValueError(f"不支持的填充模式: {pad_mode}，请选择 'left', 'right' 或 'center'")
        
    elif current_length > target_length:
        # 如果音频太长，进行截断
        if mode == 'start':
            # 从开头截取
            waveform = waveform[:, :target_length]
        elif mode == 'end':
            # 从末尾截取
            start_idx = current_length - target_length
            waveform = waveform[:, start_idx:]
        elif mode == 'center':
            # 从中间截取
            start_idx = (current_length - target_length) // 2
            waveform = waveform[:, start_idx:start_idx + target_length]
        elif mode == 'random':
            # 随机截取
            start_idx = torch.randint(0, current_length - target_length + 1, (1,)).item()
            waveform = waveform[:, start_idx:start_idx + target_length]
        else:
            raise ValueError(f"不支持的截断模式: {mode}，请选择 'start', 'center', 'end' 或 'random'")
    
    # 如果原始是单通道且输入也是单维，则恢复形状
    if num_channels == 1 and waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    
    return waveform, sample_rate
