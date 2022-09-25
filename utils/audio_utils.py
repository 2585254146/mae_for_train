# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：audio_utils.py
@IDE     ：PyCharm 
@Date    ：2021/5/19 21:59 
'''
import librosa
import numpy as np
import torch
from librosa import ParameterError
from numpy import power
from torch import nn
import torch.nn.functional as F


class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):                              # DFT正变换，可使用内置函数dftmtx(n)
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))  # 生成网格点坐标矩阵
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W


class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """
            用Conv1d（卷积神经网络）实现STFT。该函数具有librosa.core.stft的相同输出
            Conv1D (batch, steps, channels)，steps表示1篇文本中含有的单词数量，channels表示1个单词的维度
            STFT短时傅里叶变换，对一系列加窗数据做FFT(离散傅氏变换（DFT）的快速算法)。
        """
        super(STFT, self).__init__()  # 对继承自父类的属性进行初始化

        assert pad_mode in ['constant', 'reflect']  # pad_mode：填充模式

        self.n_fft = n_fft  # n_fft：窗口大小
        self.center = center
        self.pad_mode = pad_mode

        # win_length 每一帧音频都由window（）加窗。窗长win_length，然后用零填充以匹配N_FFT。默认win_length=n_fft。
        # window：字符串，元组，数字，函数 shape =（n_fft, )
        if win_length is None:
            win_length = n_fft

        # hop_length：帧移（音频样本数） 设置默认跃点（如果尚未指定）
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)  # 窗函数的调用

        # 将窗口填充到n_fft大小
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                   kernel_size=n_fft, stride=hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                   kernel_size=n_fft, stride=hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0: out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0: out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """
            input: (batch_size, data_length)
            Returns:
              real: (batch_size, n_fft // 2 + 1, time_steps)
              imag: (batch_size, n_fft // 2 + 1, time_steps)
        """
        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)  # 交换维度
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag  # 函数的实部、虚部

    #  spectrogram是一个MATLAB函数，使用短时傅里叶变换得到信号的频谱图。


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', power=2.0,
                 freeze_parameters=True):
        """
            使用pytorch计算频谱图。STFT通过Conv1d实现
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, center=center,
                         pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """
            input: (batch_size, 1, time_steps, n_fft // 2 + 1)
            Returns:
              spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram

    # 提取Log-Mel Spectrogram 特征 Log-Mel Spectrogram特征是目前在语音识别和环境声音识别中很常用的一个特征


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True,
                 ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """
            使用pytorch计算logmel频谱图。mel过滤器库是基于 librosa.filters.mel的pytorch实现
            self：要显示的矩阵
            sr：采样率
            n_fft ：FFT组件数
            频率类型：'is_log'：频谱以对数刻度显示
                    'mel'：频率由mel标度决定，
            n_mels ：产生的梅尔带数
            fmin ：最低频率（Hz）
            fmax：最高频率（以Hz为单位）
            ref ：参考值
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)

        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

        # 功率转dB

    def power_to_db(self, input):
        """
        基于librosa.core.power_to_lb实现 将功率谱（幅度平方）转换为分贝（dB）单位
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec



class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """
            Drop stripes: 在[0, drop_width]的频带范围里，随机选取stripes_num个频带置零
            Args:
              dim: int, 时间or频率的维度
              drop_width: int, 下降的最大条纹宽度
              stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim] # 时间维度/频率维度

            for n in range(batch_size): # 对每一个batch，进行
                self.transform_slice(input[n], total_width)

            return input





class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
        freq_stripes_num):
        """
        Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int,250    # 置0的宽度
          time_stripes_num: int,2    # 置0的条数
          freq_drop_width: int,63·    # 置0的宽度
          freq_stripes_num: int,2    # 置0的条数
        """
        # 输入维度为（batch_size, channel, time,fre）
        super(SpecAugmentation, self).__init__()

        # 在[0, time_drop_width]的帧数里，随机选取time_stripes_num帧置零
        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num)

        # 在[0, freq_drop_width]的频带范围里，随机选取freq_stripes_num个频带置零
        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x



def mono_to_color(X, eps=1e-6, mean=None, std=None):
    # X = np.stack([X, X, X], axis=1)
    X = X.expand(X.shape[0], 1, X.shape[2], X.shape[3])
    # X = X.numpy(X)
    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = torch.clamp(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.type(torch.float32)
    else:
        V = torch.zeros_like(X)
    return V


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size, device):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.tensor(mixup_lambdas, requires_grad=False).float().to(device)

def do_mixup(x, mixup_lambda):
    """
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out