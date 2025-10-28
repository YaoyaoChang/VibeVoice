"""!
@author Yi Luo (oulyluo) and  Jianwei Yu
"""

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, residual=True):
        super(ResRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        
        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*(int(bidirectional)+1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1,2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0], input.shape[2], input.shape[1])
        rnn_output = rnn_output.transpose(1,2).contiguous()
        
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output

class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7, num_layer=1):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(ResRNN(self.feature_dim, self.feature_dim*2, bidirectional=True))
        self.band_rnn = nn.Sequential(*self.band_rnn)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim*2, bidirectional=True)

    def forward(self, input):
        # if isinstance(input, list):
        #     input, active_nband = input[0], input[1]
        # # input shape: B, nband*N, T
        B, N, T = input.shape
        # if active_nband is None:
        band_output = self.band_rnn(input.view(B*self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0,3,2,1).contiguous().view(B*T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0,3,2,1).contiguous()
        return output.view(B, N, T)

class Separator(nn.Module):
    def __init__(self, sr=48000, win=2048, stride=512, feature_dim=128, num_layer=1, num_repeat=6, device='cuda'):
        super(Separator, self).__init__()
        
        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.device = device

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))
        self.band_width = [bandwidth_100]*10
        self.band_width += [bandwidth_250]*12
        self.band_width += [bandwidth_500]*8
        self.band_width += [bandwidth_1k]*8
        self.band_width += [bandwidth_2k]*2
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)
        
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i]*2, self.eps),
                                         nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
                                        )
                          )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband*self.feature_dim, self.nband, num_layer))             
        self.separator = nn.Sequential(*self.separator)
        
        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feature_dim, self.feature_dim*2, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*2, self.feature_dim*2, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*2, self.band_width[i]*4, 1)
                                          )
                            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
        
    def forward(self, input, sr=None):
        batch_size, nch, nsample = input.shape
        input = input.view(batch_size*nch, -1)

        # frequency-domain separation
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec_RI = []
        subband_spec_complex = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(spec_RI[:,:,band_idx:band_idx+self.band_width[i]].contiguous())
            subband_spec_complex.append(spec[:,band_idx:band_idx+self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec_RI[i].view(batch_size*nch, self.band_width[i]*2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        
        # separator
        sep_output = self.separator(subband_feature.view(batch_size*nch, self.nband*self.feature_dim, -1))  # B, nband*N, T
        if isinstance(sep_output, list):
            sep_output = sep_output[0]
        sep_output = sep_output.view(batch_size*nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:,i]).view(batch_size*nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:,0] * torch.sigmoid(this_output[:,1])  # B*nch, 2, BW, T
            this_mask_real = this_mask[:,0]  # B*nch, BW, T
            this_mask_imag = this_mask[:,1]  # B*nch, BW, T
            est_spec_real = subband_spec_complex[i].real * this_mask_real - subband_spec_complex[i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = subband_spec_complex[i].real * this_mask_imag + subband_spec_complex[i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        if spec.shape[1] > est_spec.shape[1]:
            est_spec = torch.cat([est_spec, spec[:,est_spec.shape[1]:,:]], 1)
        
        output = torch.istft(est_spec.view(batch_size*nch, self.enc_dim, -1), 
                             n_fft=self.win, hop_length=self.stride, 
                             window=torch.hann_window(self.win).to(input.device).type(input.type()), length=nsample)

        output = output.view(batch_size, nch, -1)

        return output


    @torch.no_grad()
    def process_batch(self, wav, segment = 10, batch_size=10):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        length = wav.shape[-1]
        segment_length = int(segment * self.sr)
        vocal = []
        bgm = []
        num_seg = math.ceil(length / segment_length)
        pad_length = num_seg * segment_length - length
        
        # 如果填充长度大于6秒(6*采样率)，则前面填充3秒，后面填充剩余部分
        if pad_length > 0:
            if pad_length > 6 * self.sr:
                pad_front = 3 * self.sr
                pad_end = pad_length - pad_front
                wav = torch.nn.functional.pad(wav, (pad_front, pad_end))
            else:
                # 否则仍然只在末尾填充
                wav = torch.nn.functional.pad(wav, (0, pad_length))
        
        wav = wav.view(-1, segment_length)

        num_batch = math.ceil(wav.shape[0] / batch_size)
        
        for i in tqdm(range(num_batch)):
            wav_seg = wav[int(i*batch_size):int((i+1)*batch_size)]
            wav_seg = wav_seg.to(self.device).unsqueeze(1)
            vocal_seg = self.forward(wav_seg)
            bgm_seg = wav_seg - vocal_seg

            vocal_seg = vocal_seg.squeeze(1).cpu()
            bgm_seg = bgm_seg.squeeze(1).cpu()
            vocal.append(vocal_seg)
            bgm.append(bgm_seg)
            torch.cuda.empty_cache()
        
        # 注意：由于前端也有填充，裁剪时需要考虑前端填充的长度
        if pad_length > 6 * self.sr:
            pad_front = 3 * self.sr
            vocal = torch.cat(vocal, 0).view(1, -1)[..., pad_front:pad_front+length]
            bgm = torch.cat(bgm, 0).view(1, -1)[..., pad_front:pad_front+length]
        else:
            vocal = torch.cat(vocal, 0).view(1, -1)[...,:length]
            bgm = torch.cat(bgm, 0).view(1, -1)[...,:length]
        
        return vocal, bgm

class BsrnnENHInference:
    def __init__(self, model_name, device="cuda", sample_rate=48000):
        self.model = Separator(sr=sample_rate, win=2048, stride=512, feature_dim=128, num_layer=1, num_repeat=6, device=device)
        self.model.load_state_dict(torch.load(model_name)['state_dict'])
        self.model.to(device)
        self.device = device
    
    def process_batch(self, wav, segment = 10, batch_size=10):
        return self.model.process_batch(wav, segment, batch_size)
    
if __name__ == '__main__':
    try:
        from .utils import read_audio, write_audio
    except:
        from utils import read_audio, write_audio
    
    # wav_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/data/demo/0.mp3"
    # wav_path = "/data/jianwei/experiment/AudioGeneration/msra/VibeVoice/demo/voices/en_Linda_bgm_woman.wav"
    # wav_path = "/data/jianwei/experiment/AudioGeneration/msra/VibeVoice/demo/voices/en_Tom_bgm_man.wav"
    # wav_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/exp/voice/MOTIVATIONMorganFreeman.mp3"
    # wav_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/exp/voice/oldwizard1.m4a"
    wav_path = "/data/jianwei/experiment/DataPropc/AudioAutoPrep/exp/voice/kobe1.wav"
    wav_path = "/data/yaoyaochang/code/speech/data/wangyiyun/debug/34002930.mp3"
    wav = read_audio(wav_path, sr=48000)
    model = Separator()
    ckpt = torch.load("/mnt/conversationhub/jianweiyu/DataPropc/AudioAutoPrepV2/ckpts/bsrnn/speech/bsrnnENH.pt")
    model.load_state_dict(ckpt['state_dict'])
    model.to("cuda")
    vocal, bgm = model.process_batch(wav)
    vocal = vocal.squeeze(0).numpy()
    bgm = bgm.squeeze(0).numpy()
    name = wav_path.split("/")[-1].rsplit(".", 1)[0]
    write_audio(f"{name}_vocal.mp3", vocal, sr=48000)
    write_audio(f"{name}_bgm.mp3", bgm, sr=48000)