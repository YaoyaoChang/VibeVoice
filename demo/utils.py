import numpy as np
import os
import warnings
import jiwer
import re
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union, Any
import librosa

from subprocess import CalledProcessError, run
import json


def read_audio(file_path, sr=16000, mono=True, backend="torchaudio"):
    """
    读取音频文件并返回 numpy 数组
    
    参数:
        file_path (str): 音频文件路径
        sr (int): 目标采样率，默认为 16000
        mono (bool): 是否转换为单声道，默认为 True
        backend (str): 使用的后端，可选 "torchaudio" 或 "librosa"，默认为 "torchaudio"
        
    返回:
        numpy.ndarray: 音频数据的 numpy 数组，如果mono=True则返回一维数组[采样点数]，否则返回[通道数, 采样点数]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    if backend.lower() == "torchaudio":
        try:
            import torchaudio
            import torch
            
            waveform, sample_rate = torchaudio.load(file_path)
            
            # 转换为 numpy 数组
            waveform = waveform.numpy()
            
            # 如果需要转换为单声道
            if mono and waveform.shape[0] > 1:
                # 取第一个通道而不是平均
                waveform = waveform[0]
            elif not mono and waveform.ndim == 1:
                # 如果是单声道但需要保持 [通道数, 采样点数] 的形状
                waveform = waveform[np.newaxis, :]
            
            # 重采样（如果需要）
            if sample_rate != sr:
                # 确保waveform是二维的用于重采样
                is_mono_dim = waveform.ndim == 1
                if is_mono_dim:
                    waveform = waveform[np.newaxis, :]
                
                resampler = torchaudio.transforms.Resample(sample_rate, sr)
                waveform = resampler(torch.from_numpy(waveform)).numpy()
                
                # 如果原来是一维的，恢复一维形状
                if is_mono_dim and mono:
                    waveform = waveform[0]
            if waveform.ndim > 1 and mono:
                waveform = waveform[0]


            return waveform
            
        except ImportError:
            warnings.warn("torchaudio 未安装，将使用 librosa 作为后备选项")
            backend = "librosa"
    
    if backend.lower() == "librosa":
        try:
            import librosa
            
            # librosa 默认返回 mono 音频，shape 为 [采样点数]
            waveform, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
            
            # 如果不是单声道且需要保持 [通道数, 采样点数] 的形状
            if not mono and waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            if waveform.ndim > 1 and mono:
                waveform = waveform[0]
            return waveform
            
        except ImportError:
            raise ImportError("无法导入 torchaudio 或 librosa，请安装其中至少一个库")
    
    raise ValueError(f"不支持的后端: {backend}，请使用 'torchaudio' 或 'librosa'")

def write_audio(file_path, waveform, sr=16000, mono=False, backend="torchaudio"):
    """
    将音频数据保存为文件
    
    参数:
        file_path (str): 保存音频的文件路径
        waveform (numpy.ndarray): 音频数据，形状为 [通道数, 采样点数] 或 [采样点数]
        sr (int): 采样率，默认为 16000
        mono (bool): 是否将多通道音频转换为单声道，默认为 False
        backend (str): 使用的后端，可选 "torchaudio" 或 "librosa"，默认为 "torchaudio"
    """
    # 确保目标目录存在
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # 确保输入是 numpy 数组
    if not isinstance(waveform, np.ndarray):
        raise TypeError("输入波形必须是 numpy 数组")
    
    # 处理单声道/多通道转换
    if mono and waveform.ndim > 1 and waveform.shape[0] > 1:
        # 取第一个通道
        waveform = waveform[0]
    elif waveform.ndim == 1:
        # 如果是一维数组，转换为 [1, 采样点数] 格式
        waveform = waveform[np.newaxis, :]
    
    if backend.lower() == "torchaudio":
        try:
            import torchaudio
            import torch
            
            # 转换为 torch 张量
            waveform_tensor = torch.from_numpy(waveform)
            
            # 保存音频
            torchaudio.save(file_path, waveform_tensor, sr)
            return
            
        except ImportError:
            warnings.warn("torchaudio 未安装，将使用 librosa 作为后备选项")
            backend = "librosa"
    
    if backend.lower() == "librosa":
        try:
            import soundfile as sf
            
            # librosa 本身不提供保存功能，使用 soundfile
            # 如果是多声道，需要转置为 [采样点数, 通道数] 格式
            if waveform.ndim > 1 and waveform.shape[0] <= waveform.shape[1]:
                waveform = waveform.T
            
            sf.write(file_path, waveform, sr)
            return
            
        except ImportError:
            try:
                import scipy.io.wavfile as wavfile
                
                # scipy 只支持保存为 WAV 格式
                # 如果是多声道，需要转置为 [采样点数, 通道数] 格式
                if waveform.ndim > 1 and waveform.shape[0] <= waveform.shape[1]:
                    waveform = waveform.T
                
                # scipy 需要 int16 或 float32 格式
                if waveform.dtype == np.float64:
                    waveform = waveform.astype(np.float32)
                
                wavfile.write(file_path, sr, waveform)
                return
                
            except ImportError:
                raise ImportError("无法导入 soundfile 或 scipy.io.wavfile，请安装其中至少一个库")
    
    raise ValueError(f"不支持的后端: {backend}，请使用 'torchaudio' 或 'librosa'")

def calculate_wer(reference, hypothesis):
    """
    使用 jiwer 库计算词错误率(Word Error Rate)
    
    参数:
        reference (str): 参考文本（正确文本）
        hypothesis (str): 假设文本（预测文本）
        
    返回:
        float: 词错误率，范围从0（完全匹配）到1（完全不匹配）或更高
    """
    
    
    # # 定义文本预处理转换器，保留撇号(')，去除其他标点符号
    # def normalize_text(text):
    #     # 在中文汉字之间添加空格
    #     # 使用正则表达式匹配中文字符（Unicode范围：\u4e00-\u9fff）
    #     text = re.sub(r'([\u4e00-\u9fff])(?=[\u4e00-\u9fff])', r'\1 ', text)
        
    #     # 保留撇号(')，去除其他标点符号
    #     text = re.sub(r'[^\w\s\']', '', text)
    #     # 将多个空格替换为单个空格
    #     text = re.sub(r'\s+', ' ', text)
    #     # 去除首尾空格并转换为小写
        
    #     return text.strip().lower()

    # import pdb; pdb.set_trace()
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    
    # 计算 WER
    try:
        wer = jiwer.process_words(
            reference, 
            hypothesis,
        )
    except Exception as e:
        if len(reference) == 0:
            return {
                'wer': 100,
                'ins': 100,
                'del': 0,
                'sub': 0,
                'ins_ratio': 1,
                'del_ratio': 0,
                'sub_ratio': 0,
                'is_hallucination': True
            }
        elif len(hypothesis) == 0:
            return {
                'wer': 100,
                'ins': 0,
                'del': 100,
                'sub': 0,
                'ins_ratio': 0,
                'del_ratio': 1,
                'sub_ratio': 0,
                'is_hallucination': True
            }
        else:
            return None
    
    ins_ratio = wer.insertions / len(wer.references[0])
    del_ratio = wer.deletions / len(wer.references[0])
    sub_ratio = wer.substitutions / len(wer.references[0])
    is_hallucination = (ins_ratio > 0.5 or del_ratio > 0.5) and len(wer.references[0]) > 2
    
    return {
        'wer': wer.wer,
        'ins': wer.insertions,
        'del': wer.deletions,
        'sub': wer.substitutions,
        'ins_ratio': ins_ratio,
        'del_ratio': del_ratio  ,
        'sub_ratio': sub_ratio,
        'is_hallucination': is_hallucination
    }

class SNRCalculator:
    """
    信噪比(Signal-to-Noise Ratio, SNR)计算器
    
    使用多线程处理音频片段计算信噪比
    """
    
    def __init__(self, n_threads: int = 4):
        """
        初始化信噪比计算器
        
        参数:
            n_threads (int): 用于处理的线程数，默认为10
        """
        self.n_threads = n_threads
    
    def compute_snr(self, 
                   segments: List[Dict[str, Any]], 
                   audio: np.ndarray, 
                   vocal: np.ndarray,
                   eps: float = 1e-8) -> List[Dict[str, Any]]:
        """
        计算每个音频片段的信噪比
        
        参数:
            segments (List[Dict]): 音频片段列表
            audio (np.ndarray): 原始音频数据
            vocal (np.ndarray): 提取的人声部分
            eps (float): 防止除零错误的小值
            
        返回:
            List[Dict]: 包含信噪比信息的片段列表
        """
        if len(segments) == 0:
            return segments
            
        # 确保音频和人声具有相同的形状
        if audio.shape != vocal.shape:
            raise ValueError(f"音频和人声维度不匹配: {audio.shape} vs {vocal.shape}")
            
        # 使用线程池并行处理片段
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # 为每个片段提交处理任务
            future_to_segment = {
                executor.submit(self._compute_segment_snr, segment, audio, vocal, eps): i 
                for i, segment in enumerate(segments)
            }
            
            # 收集结果并按原始顺序返回
            results = [None] * len(segments)
            for future in concurrent.futures.as_completed(future_to_segment):
                idx = future_to_segment[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"处理片段 {idx} 时出错: {e}")
                    results[idx] = segments[idx]  # 保留原始片段
                    
        return results
    
    def _compute_segment_snr(self, 
                           segment: Dict[str, Any], 
                           audio: np.ndarray, 
                           vocal: np.ndarray,
                           eps: float = 1e-8) -> Dict[str, Any]:
        """
        计算单个片段的信噪比
        
        参数:
            segment (Dict): 音频片段信息
            audio (np.ndarray): 原始音频数据
            vocal (np.ndarray): 提取的人声部分
            eps (float): 防止除零错误的小值
            
        返回:
            Dict: 添加了信噪比信息的片段
        """
        start_sample = int(segment['start_ori'] * 16000)  # 假设采样率为16kHz
        end_sample = int(segment['end_ori'] * 16000)
        
        # 确保索引在范围内
        if end_sample > len(audio):
            end_sample = len(audio)
        
        if start_sample >= end_sample:
            segment['snr'] = 0.0
            return segment
            
        # 提取片段
        segment_audio = audio[start_sample:end_sample]
        segment_vocal = vocal[start_sample:end_sample]
        
        # 计算信噪比
        try:
            snr_value = self._calculate_snr(segment_audio, segment_vocal, eps)
        except Exception as e:
            snr_value = -np.inf
        
        # 更新片段信息
        segment_copy = segment.copy()
        segment_copy['snr'] = snr_value
        
        return segment_copy
    
    def _calculate_snr(self, 
                     x: np.ndarray, 
                     s: np.ndarray, 
                     eps: float = 1e-8) -> float:
        """
        计算尺度不变信噪比 (Scale-Invariant SNR)
        
        参数:
            x (np.ndarray): 输入信号（包含噪声）
            s (np.ndarray): 参考信号（干净信号）
            eps (float): 防止除零错误的小值
            
        返回:
            float: 信噪比值（dB）
        """
        # 零均值处理
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        
        # 计算L2范数
        def l2norm(mat):
            return np.linalg.norm(mat)
        
        # 计算信噪比（dB）
        snr_value = 20 * np.log10(l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps) + eps)
        
        return snr_value



def load_audio_use_ffmpeg(file: str, resample: bool = False, target_sr: int = 16000):
    """
    Open an audio file and read as mono waveform, optionally resampling.
    Returns both the audio data and the original sample rate.

    Parameters
    ----------
    file: str
        The audio file to open
    resample: bool
        Whether to resample the audio
    target_sr: int
        The target sample rate if resampling is requested

    Returns
    -------
    A tuple containing:
    - A NumPy array with the audio waveform in float32 dtype
    - The original sample rate of the audio file
    """
    if not resample:
        # First, get the original sample rate
        cmd_probe = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file
        ]
        
        original_sr = int(run(cmd_probe, capture_output=True, check=True).stdout.decode().strip())
    else:
        original_sr = None

    # Now load the audio
    sr_to_use = target_sr if resample else original_sr
    
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr_to_use),
        "-"
    ]
    
    out = run(cmd, capture_output=True, check=True).stdout
    audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
    return audio_data, sr_to_use

def speech_json_filter(data, 
                       min_speech_duration_ratio: float = 0.6,
                       min_num_effective_speakers: int = 1,
                       max_num_effective_speakers: int = 4,
                       max_num_speakers: int = 8,
                       min_avg_turn_duration: float = 3,
                       max_avg_turn_duration: float = np.inf):
    """
    过滤语音识别结果中的非语音片段
    
    参数:
        data: Dict|list|str
    """
    if isinstance(data, str):
        json_path = data
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        json_path = None
    
    if isinstance(data, dict):
        info = data["segments_merged"]
        if data.get("audio_length", None):
            audio_length = info[-1]["end_ori"]
    elif isinstance(data, list):
        info = data
        audio_length = data[-1]["end_ori"]

    speech_duration = sum(
        segment["end_ori"] - segment["start_ori"]
        for segment in info
    )
    speech_duration_ratio = speech_duration / audio_length
    
    speaker_ratio_dict = {}
    if 'spectral_all_center_similarities' not in info[0]:
        method = 'umap'
    elif 'umap_all_center_similarities' not in info[0]:
        method = 'spectral'
    else:
        if len(info[0]['spectral_all_center_similarities']) > len(info[0]['umap_all_center_similarities']):
            num_speakers = len(info[0]['spectral_all_center_similarities'])
            method = 'spectral'
        else:
            num_speakers = len(info[0]['umap_all_center_similarities'])
            method = 'umap'

    speaker_turn_list = []
    cur_speaker = None
    for segment in info:
        speaker = segment[method + '_segment_labels']
        speaker = int(
            str(speaker).replace('_OUTLIER', '')
        )
        if speaker not in speaker_ratio_dict:
            speaker_ratio_dict[speaker] = 0
        speaker_ratio_dict[speaker] += segment["end_ori"] - segment["start_ori"]
        if cur_speaker != speaker:
            speaker_turn_list.append(speaker)
            cur_speaker = speaker
        
    for speaker, duration in speaker_ratio_dict.items():
        speaker_ratio_dict[speaker] = duration / speech_duration

    num_effective_speakers = sum(1 for ratio in speaker_ratio_dict.values() if ratio >= 0.1)
    avg_turn_duration = speech_duration / len(speaker_turn_list)

    
    meta_info = {
        "num_effective_speakers": num_effective_speakers,
        "avg_turn_duration": avg_turn_duration,
        "speaker_turn_list": speaker_turn_list,
        "speaker_ratio_dict": speaker_ratio_dict,
        "speech_duration_ratio": speech_duration_ratio,
        "audio_length": audio_length,
    }
    if json_path:
        meta_info["json_path"] = json_path
    
    if (
        speech_duration_ratio >= min_speech_duration_ratio and
        num_effective_speakers >= min_num_effective_speakers and
        num_effective_speakers <= max_num_effective_speakers and
        avg_turn_duration >= min_avg_turn_duration and
        avg_turn_duration <= max_avg_turn_duration
    ):  
        meta_info["is_valid"] = True
        return meta_info
    else:
        meta_info["is_valid"] = False
        return meta_info

def avoid_clipping(y, scalar=None, eps=1e-5):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps
        if scalar < 1:
            scalar = 1 + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y**2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar

def norm_audio(audio):
    # 能量归一化
    audio, _, scalar_dB = tailor_dB_FS(audio)
    audio, scalar_clip = avoid_clipping(audio, scalar_dB)
    return audio, scalar_dB, scalar_clip

def denormalize_audio(audio, scalar_dB, scalar_clip):
    audio *= scalar_clip
    audio *= scalar_dB
    return audio

def read_audio_segment(wav_path, start_time, end_time, sr=24000):
    """
    读取音频文件的指定时间段
    
    参数:
    wav_path: str, 音频文件路径
    start_time: float, 开始时间(秒)
    end_time: float, 结束时间(秒)
    sr: int, 采样率
    
    返回:
    audio_segment: numpy.ndarray, 音频数据
    sr: int, 采样率
    """
    # 计算持续时间
    duration = end_time - start_time
    
    # 使用librosa.load读取指定时间段
    audio_segment, sr = librosa.load(wav_path, 
                                   sr=sr,
                                   offset=start_time,
                                   duration=duration)
    
    return audio_segment, sr

def normalize_text(text):
    # 在中文汉字之间添加空格
    # 使用正则表达式匹配中文字符（Unicode范围：\u4e00-\u9fff）
    text = re.sub(r'([\u4e00-\u9fff])(?=[\u4e00-\u9fff])', r'\1 ', text)
    
    # 保留撇号(')，去除其他标点符号
    text = re.sub(r'[^\w\s\']', '', text)
    # 将多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格并转换为小写
    
    return text.strip().lower()

def count_chars_mixed_language(text):
    """
    计算中英文混合文本的字符数
    中文一个字算一个字符，英文一个单词算一个字符
    
    参数:
        text (str): 输入文本
        
    返回:
        int: 字符数
    """
    # 先规范化文本
    normalized_text = normalize_text(text)
    
    # 分割成tokens
    tokens = normalized_text.split()
    
    char_count = 0
    for token in tokens:
        # 检查是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', token):
            # 如果包含中文，每个中文字符算一个
            char_count += len(re.findall(r'[\u4e00-\u9fff]', token))
            # 加上非中文字符组成的单词数（如果有的话）
            non_chinese = re.sub(r'[\u4e00-\u9fff]', '', token)
            if non_chinese.strip():
                char_count += 1
        else:
            # 纯英文单词算一个字符
            char_count += 1
    
    return char_count

def text_audio_ratio(segments, max_ratio=10, min_ratio=1):
    """
    计算文本音频比例并判断异常
    
    参数:
        segments (list): 音频片段列表
        max_ratio (float): 最大比例阈值
        min_ratio (float): 最小比例阈值
        
    返回:
        list: 包含比例信息和异常标记的片段列表
    """
    ratio_list = []
    updated_segments = []
    
    # 计算每个片段的文本音频比例
    for segment in segments:
        dur = segment['end_ori'] - segment['start_ori']
        num_char = count_chars_mixed_language(segment['text'])
        
        # 避免除零错误
        if dur > 0:
            ratio = num_char / dur
        else:
            ratio = 0
            
        ratio_list.append(ratio)
        
        # 复制片段并添加比例信息
        updated_segment = segment.copy()
        updated_segment['text_audio_ratio'] = ratio
        updated_segment['num_chars'] = num_char
        updated_segments.append(updated_segment)
    
    # 计算所有比例的均值和标准差
    if len(ratio_list) > 0:
        mean_ratio = np.mean(ratio_list).item()
        std_ratio = np.std(ratio_list).item()
    else:
        mean_ratio = 0
        std_ratio = 0
    
    # 判断每个片段是否异常
    for i, segment in enumerate(updated_segments):
        ratio = ratio_list[i]
        
        # 判断是否异常的条件：
        # 1. 超过绝对阈值范围 (min_ratio, max_ratio)
        # 2. 超过均值的2倍或低于均值的0.5倍
        abnormal_by_threshold = ratio < min_ratio or ratio > max_ratio
        abnormal_by_mean = False
        
        if mean_ratio > 0:  # 避免除零错误
            abnormal_by_mean = ratio > mean_ratio * 2 or ratio < mean_ratio * 0.5
        
        segment['abnormal_ratio'] = abnormal_by_threshold or abnormal_by_mean
        segment['abnormal_by_threshold'] = abnormal_by_threshold
        segment['abnormal_by_mean'] = abnormal_by_mean
            
        # 添加统计信息
        segment['ratio_mean'] = mean_ratio
        segment['ratio_std'] = std_ratio
    
    return updated_segments


