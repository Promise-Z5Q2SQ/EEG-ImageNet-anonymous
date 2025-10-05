import numpy as np
import mne
from utilities import *


def de_feat_cal_S1(eeg_data, subject, granularity):
    if os.path.exists(os.path.join('../data/de_feat_S1/', f"{subject}_{granularity}_de.npy")):
        return np.load(os.path.join('../data/de_feat_S1/', f"{subject}_{granularity}_de.npy"))
    else:
        channel_names = [f'EEG{i}' for i in range(1, 63)]
        info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
        _epochs = mne.EpochsArray(data=eeg_data, info=info)
        de_feat_list = []
        for f_min, f_max in FREQ_BANDS.values():
            spectrum = _epochs.compute_psd(fmin=f_min, fmax=f_max)
            psd = spectrum.get_data() + 1e-10
            diff_entropy = np.sum(np.log(psd), axis=-1)
            de_feat_list.append(diff_entropy)
        _de_feat = np.concatenate(de_feat_list, axis=1)
        # print(_de_feat.shape)  # de_feat.shape = (4000, 310), normally
        np.save(os.path.join('../data/de_feat_S1/', f"{subject}_{granularity}_de.npy"), _de_feat)
        return _de_feat


def de_feat_cal_S2(eeg_data, subject, granularity, stage):
    if stage == 30:
        save_dir = "../data/de_feat_S2/30/"
    elif stage == 20:
        save_dir = "../data/de_feat_S2/20/"
    else:
        de_feat_30 = de_feat_cal_S2(eeg_data, subject, granularity, 30)
        de_feat_20 = de_feat_cal_S2(eeg_data, subject, granularity, 20)
        return np.concatenate([de_feat_30, de_feat_20], axis=0)
    if os.path.exists(os.path.join(save_dir, f"{subject}_{granularity}_de.npy")):
        return np.load(os.path.join(save_dir, f"{subject}_{granularity}_de.npy"))
    else:
        channel_names = [f"EEG{i}" for i in range(1, 63)]
        info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types="eeg")
        _epochs = mne.EpochsArray(data=eeg_data, info=info)
        de_feat_list = []
        for f_min, f_max in FREQ_BANDS.values():
            spectrum = _epochs.compute_psd(fmin=f_min, fmax=f_max)
            psd = spectrum.get_data() + 1e-10
            diff_entropy = np.sum(np.log(psd), axis=-1)
            de_feat_list.append(diff_entropy)
        _de_feat = np.concatenate(de_feat_list, axis=1)
        np.save(os.path.join(save_dir, f"{subject}_{granularity}_de.npy"), _de_feat)
        return _de_feat
