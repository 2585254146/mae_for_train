# -*- coding: UTF-8 -*-
'''
@Author  ：YujieZhong
@File    ：audio_augmentation.py
@IDE     ：PyCharm 
@Date    ：2021/9/14 21:09 
'''
import glob

import librosa, librosa.display
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import colorednoise as cn




class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):     # 通过增加__callable__()函数，将类实例化对象变为可调用
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


class AddGaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_amplitude=0.5, **kwargs):
        super().__init__(always_apply, p)

        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, y: np.ndarray, **params):
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_amplitude).astype(y.dtype)
        return augmented

class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=10.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=10.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

class BrownNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=10.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(2, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class AddCustomNoise(AudioTransform):
    """
    This Function allows you to add noise from any custom file you want just give path to the directory where the files
    are stored and you are good to go.
    """

    def __init__(self, file_dir, always_apply=False, p=0.5):
        super(AddCustomNoise, self).__init__(always_apply, p)
        '''
        file_dir must be of form '.../input/.../something'
        '''

        self.noise_files = glob.glob(file_dir + '/*')

    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''
        nf = self.noise_files[int(np.random.uniform(0, len(self.noise_files)))]

        noise, _ = librosa.load(nf)

        if len(noise) > len(data):
            start_ = np.random.randint(len(noise) - len(data))
            noise = noise[start_: start_ + len(data)]
        else:
            noise = np.pad(noise, (0, len(data) - len(noise)), "constant")

        data_wn = data + noise

        return data_wn


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_steps=5, sr=32000):
        super().__init__(always_apply, p)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented

class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1.2):
        super().__init__(always_apply, p)

        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, p)

        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


class PolarityInversion(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PolarityInversion, self).__init__(always_apply, p)

    def apply(self, data, **params):
        '''
        data : ndarray of audio timeseries
        '''
        return -data

class Gain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False, p=0.5):
        super(Gain, self).__init__(always_apply, p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db


    def apply(self, data, **args):
        amplitude_ratio = 10**(np.random.uniform(self.min_gain_in_db, self.max_gain_in_db)/20)
        return data * amplitude_ratio


class spec_augment(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, num_mask=2,
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.2):
        super().__init__(always_apply, p)
        self.num_mask = num_mask
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.time_masking_max_percentage = time_masking_max_percentage

    def apply(self, y: np.ndarray, **params):
        "y: spectrumgram"

        for i in range(self.num_mask):
            all_frames_num, all_freqs_num = y.shape
            freq_percentage = np.random.uniform(0.0, self.freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            y[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = np.random.uniform(0.0, self.time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            y[t0:t0 + num_frames_to_mask, :] = 0

        return y



def wave_aug(p: float):
    return Compose([
      OneOf([
        GaussianNoiseSNR(p=p),
        PinkNoiseSNR(p=p),
        BrownNoiseSNR(p=p),
        # AddCustomNoise(p=p)
      ]),
      PitchShift(p=p),
      TimeShift(p=p),
      VolumeControl(mode="sine", p=p),
      Gain(p=p)
    ])

def spec_aug(p: float):
    return Compose([spec_augment(p=p,
                                 num_mask=2,
                                 freq_masking_max_percentage=0.15,
                                 time_masking_max_percentage=0.25)])


if __name__ == '__main__':
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    # fig.suptitle('Audioaugment', fontsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图间距
    plt.tight_layout(3)
    wav_path = r"C:\Users\LQR\Desktop\bird\feature_show\test_wav\北灰鹟.wav"
    wav,sr = librosa.load(wav_path, sr=32000)

    subplot(2, 3, 1)
    aug = GaussianNoiseSNR(p=1)
    wav1 = aug(wav)
    melspec = librosa.feature.melspectrogram(wav1, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspecDB = librosa.power_to_db(melspec)
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("added noise", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')

    subplot(2, 3, 2)
    aug = PitchShift(p=1)
    wav2 = aug(wav)
    melspec = librosa.feature.melspectrogram(wav2, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspecDB = librosa.power_to_db(melspec)
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("pitch shift", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')

    subplot(2, 3, 3)
    aug = TimeShift(p=1)
    wav3 = aug(wav)
    melspec = librosa.feature.melspectrogram(wav3, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspecDB = librosa.power_to_db(melspec)
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("time shift", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')

    subplot(2, 3, 4)
    aug = Gain(p=1)
    wav4 = aug(wav)
    melspec = librosa.feature.melspectrogram(wav4, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspecDB = librosa.power_to_db(melspec)
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("Gain", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')
    #
    # # subplot(2, 2, 4)
    # # aug = wave_aug(p=1)
    # # # spec = spec_aug(p=1)
    # # wav5 = aug(wav)
    # # melspec = librosa.feature.melspectrogram(wav5, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    # # # melspec = spec(melspec)
    # # melspecDB = librosa.power_to_db(melspec)
    # # # melspecDB[10:15, 30:35] = -60
    # # # melspecDB[6:11, 10:15] = -60
    # # # melspecDB[2:7, 40:45] = -60
    # # librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    # # title("time-domain augment", fontsize=7)
    # # cb = plt.colorbar()
    # # cb.set_label('Amplititude/dB')
    #
    # show()

    # fig, ax = plt.subplots(2, 2, figsize=(9, 12))
    # fig.suptitle('TimeFrequency-domain Audioaugment', fontsize=16)
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)  # 调整子图间距

    wav_path = r"C:\Users\LQR\Desktop\bird\feature_show\test_wav\北灰鹟.wav"
    wav,sr = librosa.load(wav_path, sr=32000)

    subplot(2, 3, 5)
    spec = spec_augment(p=1,
                       num_mask=2,
                       freq_masking_max_percentage=0.15,
                       time_masking_max_percentage=0.25)
    melspec = librosa.feature.melspectrogram(wav, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspec = spec(melspec)
    melspecDB = librosa.power_to_db(melspec,ref=0.5)
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("TimeFrequency-Maskchannel", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')

    # subplot(3, 2, 6)
    # melspec = librosa.feature.melspectrogram(wav, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    # melspecDB = librosa.power_to_db(melspec)
    # melspecDB[60:110, 10:25] = -60
    # melspecDB[25:75, 10:40] = -60
    # melspecDB[40:90, 40:90] = -60
    # librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    # title("Randomarea dropout", fontsize=12)
    # cb = plt.colorbar()
    # cb.set_label('Amplititude/dB')


    subplot(2, 3, 6)
    aug = wave_aug(p=1)
    spec = spec_aug(p=1)
    wav5 = aug(wav)
    melspec = librosa.feature.melspectrogram(wav5, sr=32000, n_mels=128, n_fft=1024, hop_length=512, win_length=1024)
    melspec = spec(melspec)
    melspecDB = librosa.power_to_db(melspec,ref=0.5)
    # melspecDB[10:15, 30:35] = -60
    # melspecDB[6:11, 10:15] = -60
    # melspecDB[2:7, 40:45] = -60
    librosa.display.specshow(melspecDB, x_axis="time", y_axis="mel")
    title("Randomset_Augment", fontsize=12)
    cb = plt.colorbar()
    cb.set_label('Amplititude/dB')

    show()

