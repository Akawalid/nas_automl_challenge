import os
import sys
import time
from collections import OrderedDict

import mne
import numpy as np
from Alignments.riemannian import compute_riemannian_alignment
from imblearn.under_sampling import RandomUnderSampler
from pyriemann.estimation import XdawnCovariances

# from moabb.paradigms import CVEP
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "D:/s.velut/Documents/These\\moabb\\moabb\\datasets")
sys.path.insert(0, "D:/s.velut/Documents/These\\moabb\\moabb\\paradigms")
sys.path.insert(0, "D:/s.velut/Documents/These\\Protheus_PHD\\Scripts")
sys.path.insert(0, "D:/s.velut/Documents/These\\riemannian_tSNE")
from R_TSNE import R_TSNE

# from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40
__author__ = "Sebastien Velut"


def to_window_cov(
    data,
    labels,
    codes,
    normalise=False,
    window_size=0.25,
    fps=60,
    sfreq=500,
    n_channels=32,
    estimator="lwf",
):
    n_samples_windows = int(window_size * sfreq)
    length = int((2.2 - window_size) * sfreq)
    X = np.empty(shape=((length) * data.shape[0], data.shape[1], n_samples_windows))
    y = np.empty(shape=((length) * data.shape[0]), dtype=int)
    count = 0
    for trial_nb, trial in enumerate(data):
        lab = labels[trial_nb]
        c = codes[lab]
        code_pos = 0
        for idx in range(length):
            X[count] = trial[:, idx : idx + n_samples_windows]
            if idx / sfreq >= (code_pos + 1) / fps:
                code_pos += 1
            y[count] = int(c[code_pos])
            count += 1
    if normalise:
        X_std = X.std(axis=0)
        X /= X_std + 1e-8
    xdawncov = XdawnCovariances(estimator=estimator, xdawn_estimator=estimator, nfilter=8)
    X = xdawncov.fit_transform(X, y)
    # X = X.astype(np.float32)
    y_pred = np.vstack((y, np.abs(1 - y))).T
    y = np.array([1 if (y >= 0.5) else 0 for y in y_pred[:, 0]])
    return X, y


def to_window_old(
    data,
    labels,
    length,
    n_samples_windows,
    codes,
    window_size=0.25,
    sfreq=500,
    fps=60,
    n_channels=8,
):
    X = np.empty(shape=((length) * data.shape[0], n_channels, n_samples_windows))
    idx_taken = []
    y = np.empty(shape=((length) * data.shape[0]), dtype=int)
    count = 0
    for trial_nb, trial in enumerate(data):
        lab = labels[trial_nb]
        c = codes[lab]
        code_pos = 0
        for idx in range(length):
            X[count] = trial[:, idx : idx + n_samples_windows]
            if idx / sfreq >= (code_pos + 1) / fps:
                code_pos += 1
            y[count] = int(c[code_pos])
            count += 1

        for idx in range(length):
            idx_taken.append(trial_nb * length + idx)

    y_pred = np.vstack((y, np.abs(1 - y))).T
    y = np.array([1 if (y >= 0.5) else 0 for y in y_pred[:, 0]])

    return X, y, np.array(idx_taken)


def data_train_by_parti(
    ind2take,
    raw_data,
    labels,
    on_frame=True,
    toSPD=False,
    recenter=False,
    codes=None,
    normalise=False,
    window_size=0.25,
    fps=60,
    sfreq=500,
    n_channels=32,
    estimator="lwf",
):
    X_train = []
    Y_train = []
    domains = []
    for k in range(len(ind2take)):
        if not on_frame:
            n_samples_windows = int(window_size * sfreq)
            length = int((2.2 - window_size) * sfreq)
            if toSPD:
                if normalise:
                    X_std = raw_data[k].std(axis=0)
                    temp_X /= X_std + 1e-8
                else:
                    temp_X = raw_data[k]
                temp_X, temp_Y = to_window_cov(
                    temp_X,
                    labels[k],
                    codes,
                    window_size,
                    fps,
                    sfreq,
                    n_channels,
                    estimator,
                )
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(
                        temp_X, mean=None, dtype="covmat"
                    )
            else:
                if normalise:
                    X_std = raw_data[k].std(axis=0)
                    temp_X /= X_std + 1e-8
                else:
                    temp_X = raw_data[k]
                temp_X, temp_Y, _ = to_window_old(
                    temp_X,
                    labels[k],
                    length,
                    n_samples_windows,
                    codes,
                    window_size,
                    sfreq,
                    fps,
                    n_channels,
                )
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(temp_X, mean=None, dtype="real")

        else:
            if toSPD:
                temp_X = Euc2SPD(raw_data[k], labels[k], normalise, estimator)
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(
                        temp_X, mean=None, dtype="covmat"
                    )
                temp_Y = labels[k]
            else:
                temp_X = raw_data[k].copy()
                if normalise:
                    X_std = temp_X.std(axis=0)
                    temp_X /= X_std + 1e-8
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(temp_X, mean=None, dtype="real")
                temp_Y = labels[k].copy()
        X_train.append(temp_X)
        Y_train.append(temp_Y)
        domains.append(
            [
                "Source_sub_{}".format(ind2take[k]),
            ]
            * len(temp_Y)
        )
    return np.array(X_train), np.array(Y_train), np.array(domains)


def get_codes(event_id):
    from collections import OrderedDict

    codes = OrderedDict()
    for k, v in event_id.items():
        code = k.split("_")[0]
        code = code.replace(".", "").replace("2", "")
        idx = k.split("_")[1]
        codes[v - 1] = np.array(list(map(int, code)))

    return codes


# def get_BVEP_data(subject,on_frame=True,to_keep=None,moabb_ds=CasitllosBurstVEP100(),window_size=0.25,start=-0.01,l_freq=1,h_freq=25):
#     """
#     subject : list of subject
#     """
#     if on_frame:
#         return get_BVEP_data_on_frame(subject,to_keep,moabb_ds,window_size,start,l_freq,h_freq)
#     else:
#         return get_BVEP_data_on_sample(subject,to_keep)


# def get_BVEP_data_on_frame(subject,to_keep=None,moabb_ds=CasitllosBurstVEP100(),window_size=0.25,start=-0.01,l_freq=1,h_freq=25):
#     dataset_moabb = moabb_ds
#     paradigm = CVEP()
#     print(paradigm.n_classes)

#     raw = dataset_moabb.get_data(subjects=subject)

#     raw_data = []
#     keys = list(raw.keys())
#     labels = []
#     labels_code = []
#     print(subject)
#     print(keys)

#     for ind_i,i in enumerate(subject):
#         i-=1
#         temp = raw[keys[ind_i]]["0"]["0"]

#         temp_raw = temp.copy()
#         print(temp_raw.ch_names)
#         trial_chan = temp_raw.pick_channels(["stim_trial"],verbose=False)
#         data = trial_chan.get_data()[0]
#         labels_code.append(np.array(list(filter(lambda num: num != 0, data)),dtype=int)-200)

#         if to_keep is not None:
#             temp = temp.drop_channels([ch for ch in temp.ch_names if ch not in to_keep])

#         temp = temp.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=True)
#         mne.set_eeg_reference(temp, 'average', copy=False, verbose=False)
#         events = mne.find_events(temp,["stim_epoch"])
#         epochs = mne.Epochs(temp,events,{"0":100,"1":101},picks=temp.ch_names[:-2],tmin=start,tmax=start+window_size)
#         labels.append(epochs.events[...,-1]-100)
#         raw_data.append(epochs.get_data())

#     raw_data = np.array(raw_data)
#     labels = np.array(labels)
#     labels_code = np.array(labels_code)

#     return raw_data,labels,dataset_moabb.codes,labels_code


def get_BVEP_data_on_sample(subject, to_keep=None):
    raw_data_dl = []
    labels_dl = []
    n_channels = 32

    for i in subject:
        path = "/".join(
            ["D:/s.velut/Documents/These\\Protheus_PHD\\Class4", "P" + str(i)]
        )
        file_name = "_".join(["P" + str(i), "burst100.set"])
        raw_i = mne.io.read_raw_eeglab(
            os.path.join(path, file_name), preload=True, verbose=False
        )
        if to_keep is not None:
            raw_i = raw_i.drop_channels(
                [ch for ch in raw_i.ch_names if ch not in to_keep]
            )
        raw_i = raw_i.filter(l_freq=50.1, h_freq=49.9, method="iir", verbose=True)
        mne.set_eeg_reference(raw_i, "average", copy=False, verbose=False)

        events, event_id = mne.events_from_annotations(
            raw_i, event_id="auto", verbose=False
        )
        to_remove = []
        for idx in range(len(raw_i.annotations.description)):
            if (
                ("collects" in raw_i.annotations.description[idx])
                or ("iti" in raw_i.annotations.description[idx])
                or (raw_i.annotations.description[idx] == "[]")
            ):
                to_remove.append(idx)
            else:
                code = raw_i.annotations.description[idx].split("_")[0]
                lab = raw_i.annotations.description[idx].split("_")[1]
                code = code.replace("\n", "")
                code = code.replace("[", "")
                code = code.replace("]", "")
                code = code.replace(" ", "")
                raw_i.annotations.description[idx] = code + "_" + lab

        to_remove = np.array(to_remove)
        if len(to_remove) > 0:
            raw_i.annotations.delete(to_remove)
        # Get the events
        events, event_id = mne.events_from_annotations(
            raw_i, event_id="auto", verbose=False
        )
        shift = 0.0
        # Epoch the data following event
        epochs = mne.Epochs(
            raw_i,
            events,
            event_id=event_id,
            tmin=shift,
            tmax=2.2 + shift,
            baseline=(None, None),
            preload=False,
            verbose=False,
        )
        label = epochs.events[..., -1]
        label -= np.min(label)
        labels_dl.append(label)
        data = epochs.get_data()
        info_ep = epochs.info

        raw_data_dl.append(data)

    raw_data_dl = np.array(raw_data_dl)
    labels_dl = np.array(labels_dl)
    codes = get_codes(event_id)

    return raw_data_dl, labels_dl, codes, labels_dl


def Euc2SPD(X, y, normalise=False, estimator="lwf"):
    if normalise:
        X_std = X.std(axis=0)
        X /= X_std + 1e-8

    xdawncov = XdawnCovariances(
        estimator=estimator, xdawn_estimator=estimator, nfilter=16, classes=[1]
    )
    X = xdawncov.fit_transform(X, y)
    # print("shape of cov",X.shape)

    return X


def prepare_data(
    subject,
    raw_data,
    labels,
    on_frame,
    toSPD,
    recenter=False,
    codes=None,
    normalise=False,
    window_size=0.25,
    fps=60,
    sfreq=500,
    n_channels=32,
    estimator="lwf",
):
    if on_frame:
        return prepare_data_on_frame(
            np.array(subject) - 1,
            raw_data,
            labels,
            toSPD,
            recenter,
            None,
            normalise,
            window_size,
            fps,
            sfreq,
            raw_data.shape[-2],
            estimator,
        )
    else:
        return prepare_data_on_sample(
            np.array(subject) - 1,
            raw_data,
            labels,
            toSPD,
            recenter,
            codes,
            normalise,
            window_size,
            fps,
            sfreq,
            raw_data.shape[-2],
            estimator,
        )


def prepare_data_on_frame(
    subject,
    raw_data,
    labels,
    toSPD,
    recenter,
    codes,
    normalise,
    window_size,
    fps,
    sfreq,
    n_channels,
    estimator,
):
    X, Y, domains = data_train_by_parti(
        subject,
        raw_data,
        labels,
        True,
        toSPD,
        recenter,
        codes,
        normalise,
        window_size,
        fps,
        sfreq,
        n_channels,
        estimator,
    )

    return X, Y, domains


def prepare_data_on_sample(
    subject,
    raw_data,
    labels,
    toSPD,
    recenter,
    codes,
    normalise,
    window_size,
    fps,
    sfreq,
    n_channels,
    estimator,
):
    X, Y, domains = data_train_by_parti(
        subject,
        raw_data,
        labels,
        False,
        toSPD,
        recenter,
        codes,
        normalise,
        window_size,
        fps,
        sfreq,
        n_channels,
        estimator,
    )

    return X, Y, domains


def preproc_prepare(
    subjects,
    subtest,
    raw_data_train,
    raw_data_test,
    labels_train,
    labels_test,
    modes,
    toSPD,
    recenter,
    codes,
    normalise,
    window_size,
    fps,
    sfreq,
    n_channels,
    estimator,
):
    if "DA" in modes:
        return 0


def __preproc_preparation(
    ind2take,
    raw_data,
    labels,
    on_frame=True,
    toSPD=False,
    recenter=False,
    codes=None,
    normalise=False,
    window_size=0.25,
    fps=60,
    sfreq=500,
    n_channels=32,
    estimator="lwf",
):
    X_train = []
    Y_train = []
    domains = []
    for k in range(len(ind2take)):
        if not on_frame:
            if toSPD:
                if normalise:
                    X_std = raw_data[k].std(axis=0)
                    raw_data[k] /= X_std + 1e-8
                temp_X, temp_Y = to_window_cov(
                    raw_data[k],
                    labels[k],
                    codes,
                    window_size,
                    fps,
                    sfreq,
                    n_channels,
                    estimator,
                )
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(
                        temp_X, mean=None, dtype="covmat"
                    )
            else:
                if normalise:
                    X_std = raw_data[k].std(axis=0)
                    raw_data[k] /= X_std + 1e-8
                temp_X, temp_Y, _ = to_window_old(
                    raw_data[k], labels[k], codes, window_size, fps, sfreq, n_channels
                )
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(temp_X, mean=None, dtype="real")

        else:
            if toSPD:
                temp_X = Euc2SPD(raw_data[k], labels[k], normalise, estimator)
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(
                        temp_X, mean=None, dtype="covmat"
                    )
                temp_Y = labels[k]
            else:
                if normalise:
                    X_std = raw_data[k].std(axis=0)
                    raw_data[k] /= X_std + 1e-8
                temp_X = raw_data[k]
                if recenter:
                    print("Recentering the matrix")
                    temp_X = compute_riemannian_alignment(temp_X, mean=None, dtype="real")
                temp_Y = labels[k]
        X_train.append(temp_X)
        Y_train.append(temp_Y)
        domains.append(
            [
                "Source_sub_{}".format(ind2take[k]),
            ]
            * len(temp_Y)
        )
    return np.array(X_train), np.array(Y_train), np.array(domains)


def balance(X, Y, domains):
    X_new = []
    Y_new = []
    domains_new = []
    if domains is not None:
        for d in np.unique(domains):
            ind_domain = np.where(domains == d)
            rus = RandomUnderSampler()
            counter = np.array(range(0, len(Y[ind_domain]))).reshape(-1, 1)
            index, _ = rus.fit_resample(counter, Y[ind_domain])
            index = np.sort(index, axis=0)
            X_new.append(np.squeeze(X[ind_domain][index, :, :], axis=1))
            Y_new.append(np.squeeze(Y[ind_domain][index]))
            domains_new.append(np.squeeze(domains[ind_domain][index]))
        return np.concatenate(X_new), np.concatenate(Y_new), np.concatenate(domains_new)
    else:
        rus = RandomUnderSampler()
        counter = np.array(range(0, len(Y))).reshape(-1, 1)
        index, _ = rus.fit_resample(counter, Y)
        index = np.sort(index, axis=0)
        X = np.squeeze(X[index, :, :], axis=1)
        Y = np.squeeze(Y[index])
        return X, Y, None


def get_score(clf, X_train, Y_train, X_test, Y_test):
    covs_train, x_val, y_train, y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42, shuffle=True
    )
    clf.fit(
        covs_train,
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=(np.array(x_val), y_val),
        shuffle=True,
    )

    y_pred = clf.predict(X_test)[:]

    y_test = np.array([Y_test == i for i in np.unique(Y_test)]).T
    y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return roc_auc_score(y_test, y_pred)


def get_y_pred(clf, X_train, Y_train, X_test, Y_test):
    covs_train, x_val, y_train, y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42, shuffle=True
    )
    start = time.time()
    clf.fit(
        covs_train,
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=(np.array(x_val), y_val),
        shuffle=True,
    )
    tps_train = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)[:]
    tps_test = time.time() - start

    return y_pred, tps_train, tps_test


def get_TSNE_visu(X_to_visu, shape):
    R_TSNE_moabb = R_TSNE(
        perplexity=int(1.5 * shape), verbosity=1, max_it=360, max_time=3600
    )
    res_tSNE_moabb_mat = R_TSNE_moabb.fit(X_to_visu)

    return res_tSNE_moabb_mat


########### New dataset


def onset_anno(
    onset_window,
    label_window,
    onset_code,
    nb_seq_min,
    nb_seq_max,
    code_freq,
    sfreq,
    win_size,
):
    assert sfreq != 0
    new_onset = []
    new_onset_0 = []
    current_code = 0
    onset_code = np.ceil(onset_code * code_freq / sfreq)
    nb_seq_min -= 1
    onset_shift = onset_code[current_code + nb_seq_min]
    time_trial = 2.2 - win_size
    # onset_window = np.arange(0,time_trial*code_freq*(nb_seq_max-nb_seq_min)-1,1,dtype=int)
    for i, o in enumerate(onset_window):
        if label_window[i] == 1:
            # print(i)
            if current_code == nb_seq_max - 1 - nb_seq_min:
                new_onset.append(o + onset_shift)
            else:
                if (
                    o + onset_shift
                    >= onset_code[current_code + nb_seq_min] + time_trial * code_freq
                ):
                    current_code += 1
                    onset_shift = (
                        onset_code[current_code + nb_seq_min]
                        - time_trial * code_freq * current_code
                    )
                new_onset.append(o + onset_shift)
        else:
            if current_code == nb_seq_max - 1 - nb_seq_min:
                new_onset_0.append(o + onset_shift)
            else:
                if (
                    o + onset_shift
                    >= onset_code[current_code + nb_seq_min] + time_trial * code_freq
                ):
                    current_code += 1
                    onset_shift = (
                        onset_code[current_code + nb_seq_min]
                        - time_trial * code_freq * current_code
                    )
                new_onset_0.append(o + onset_shift)

    # modified_onset_code = [onset_code[i]-time_trial*sfreq*i for i in range(nb_seq_min,nb_seq_max)]
    # new_onset_0 = np.concatenate([np.arange(onset_code[i],onset_code[i]+time_trial*sfreq,sfreq//60) for i in range(nb_seq_min,nb_seq_max)])
    new_onset_0 = np.array(list(filter(lambda i: i not in new_onset, new_onset_0)))
    # print(new_onset_0.shape)
    return np.array(new_onset) / code_freq, np.array(new_onset_0) / code_freq


def get_new_dataset(path, subject, window_size=0.35, recenter=True, fmin=1, fmax=45):
    n_class = 5
    fps = 60
    sfreq = 500
    nb_subject = len(subject)

    raw_eeglab = [
        mne.io.read_raw_eeglab(
            os.path.join(path, "_".join(["P" + str(subject[i]), "dryburst100.set"])),
            preload=True,
            verbose=False,
        )
        .resample(sfreq=sfreq)
        .filter(l_freq=fmin, h_freq=fmax, method="fir", verbose=True)
        for i in range(len(subject))
    ]

    epochs_list = []
    events_list = []
    events_id_list = []
    onset_code_list = []
    data_list = []
    labels_code_list = []
    for ind_i, i in enumerate(subject):
        # Strip the annotations that were script to make them easier to process
        events, event_id = mne.events_from_annotations(
            raw_eeglab[ind_i], event_id="auto", verbose=False
        )
        to_remove = []
        for idx in range(len(raw_eeglab[ind_i].annotations.description)):
            if ("boundary" in raw_eeglab[ind_i].annotations.description[idx]) or (
                "BURST" in raw_eeglab[ind_i].annotations.description[idx]
            ):
                to_remove.append(idx)

        to_remove = np.array(to_remove)
        if len(to_remove) > 0:
            raw_eeglab[ind_i].annotations.delete(to_remove)
        # Get the events
        temp_event, temp_event_id = mne.events_from_annotations(
            raw_eeglab[ind_i], event_id="auto", verbose=False
        )
        events_list.append(temp_event)
        events_id_list.append(temp_event_id)
        shift = 0.0
        # Epoch the data following event
        epochs_list.append(
            mne.Epochs(
                raw_eeglab[ind_i],
                events_list[ind_i],
                event_id=events_id_list[ind_i],
                tmin=shift,
                tmax=2.2 + shift,
                baseline=(None, None),
                preload=False,
                verbose=False,
            )
        )
        # print(events_list[ind_i])

        labels_code_list.append(epochs_list[ind_i].events[..., -1])
        labels_code_list[ind_i] -= np.min(labels_code_list[ind_i])
        # print(epochs.events[..., -1])
        data_list.append(epochs_list[ind_i].get_data())
        info_ep = epochs_list[ind_i].info
        # print(epochs)
        onset_code_list.append(epochs_list[ind_i].events[..., 0])
    data_list = np.array(data_list)

    codes = OrderedDict()
    for k, v in events_id_list[0].items():
        code = k.split("_")[0]
        code = code.replace(".", "").replace("2", "")
        idx = k.split("_")[1]
        codes[v - 1] = np.array(list(map(int, code)))

    n_samples_windows = int(window_size * sfreq)
    length = int((2.2 - window_size) * fps)
    Y = np.zeros((data_list.shape[0], length * data_list.shape[1]))
    idx_taken = np.zeros((data_list.shape[0], length * data_list.shape[1]))
    print(Y.shape)
    print(X.shape)
    for ind_i, i in enumerate(subject):
        _, Y[ind_i], idx_taken[ind_i] = to_window_old(
            data_list[ind_i],
            labels_code_list[ind_i],
            length,
            n_samples_windows,
            codes,
            window_size=window_size,
            sfreq=fps,
        )

    for ind_i, i in enumerate(subject):
        onset, onset_0 = onset_anno(
            idx_taken[ind_i],
            Y[ind_i],
            onset_code_list[ind_i],
            1,
            n_class * 15,
            fps,
            sfreq,
            window_size,
        )
        anno = mne.Annotations(onset, 0.001 * sfreq // 60, "1")
        anno.append(onset_0, 0.001 * sfreq // 60, "0")

        raw_eeglab[ind_i] = raw_eeglab[ind_i].set_annotations(anno)

    X_parent = np.zeros(
        (
            data_list.shape[0],
            length * data_list.shape[1],
            data_list.shape[2] * 2,
            int(window_size * sfreq),
        )
    )
    domains_parent = []
    Y_parent = np.zeros((data_list.shape[0], length * data_list.shape[1]))
    domains_parent = np.zeros((data_list.shape[0], length * data_list.shape[1]))
    for ind_i, i in enumerate(subject):
        events, event_id = mne.events_from_annotations(raw_eeglab[ind_i])
        epochs = mne.Epochs(
            raw_eeglab[ind_i],
            events,
            event_id,
            tmin=0.0,
            tmax=window_size,
            baseline=(0, 0),
        )
        X_parent[i] = epochs.get_data()[:, :, :-1]
        Y_parent[ind_i] = epochs.events[..., -1] - 1
        domains_parent[ind_i] = np.array(
            [
                "Source_sub_{}".format(ind_i + 1),
            ]
            * len(Y_parent[ind_i])
        )

    return X_parent, Y_parent, domains_parent
