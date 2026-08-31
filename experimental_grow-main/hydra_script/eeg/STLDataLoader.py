# mme: 1.11 mais a priori toute version fonctionne
import os
from collections import OrderedDict

import mne
import numpy as np

__author__ = "Sebastien Velut"


class STLDataLoader:
    def __init__(
        self,
        path,
        fmin,
        fmax,
        window_size,
        sample_freq,
        fps,
        timewise,
        participants,
        n_class,
    ) -> None:
        super(STLDataLoader, self).__init__()

        self.path = path
        self.fmin = fmin
        self.fmax = fmax
        self.window_size = window_size
        self.sfreq = sample_freq
        self.fps = fps
        self.participants = participants
        self.n_class = n_class

        self.freqwise = 0
        if timewise == "frame":
            self.freqwise = fps
        elif timewise == "time_sample":
            self.freqwise = sample_freq

        self.nb_subjects = len(participants)
        self.n_samples_windows = int(window_size * self.sfreq)
        self.length = int((2.2 - window_size) * self.freqwise)

    def _changeEventID(self, events, event_id):
        new_dic = {}
        for k in event_id.keys():
            new_dic[k.split("_")[1]] = event_id[k]
            event_id[k] = int(k.split("_")[1])
        for i in range(len(events)):
            events[i][2] = int(new_dic[str(events[i][2])])

        return events, event_id

    def to_window(
        self,
        data,
        labels,
        length,
        n_samples_windows,
        codes,
        window_size=0.25,
        normalise=True,
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
                # print('Xidx:', trial_nb*length+idx, "Tidxm:", idx, 'TidxM:', idx +
                #       n_samples_windows, 'Ltrial', trial[:, idx:idx+n_samples_windows].shape)
                idx_taken.append(trial_nb * length + idx)
        # if normalise:
        #     X_std = X.std(axis=0)
        #     X /= X_std + 1e-8
        y_pred = np.vstack((y, np.abs(1 - y))).T
        y = np.array([1 if (y >= 0.5) else 0 for y in y_pred[:, 0]])

        return X, y, np.array(idx_taken)

    def onset_anno(
        self,
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
        for i, o in enumerate(onset_window):
            if label_window[i] == 1:
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

        new_onset_0 = np.array(list(filter(lambda i: i not in new_onset, new_onset_0)))
        return np.array(new_onset) / code_freq, np.array(new_onset_0) / code_freq

    def load_data(
        self,
    ):
        raw_eeglab = [
            mne.io.read_raw_eeglab(
                os.path.join(
                    self.path, "_".join([self.participants[i], "dryburst100.set"])
                ),
                preload=True,
                verbose=False,
            ).resample(sfreq=self.sfreq)
            for i in range(len(self.participants))
        ]
        raw_eeglab = [
            raw_eeglab[i].filter(
                l_freq=self.fmin, h_freq=self.fmax, method="fir", verbose=False
            )
            for i in range(len(self.participants))
        ]
        return raw_eeglab

    def get_epochs(self, raw_data, shift=0.0):
        epochs_list = []
        events_list = []
        events_id_list = []
        onset_code_list = []
        data_list = []
        labels_code_list = []
        for ind_i, i in enumerate(self.participants):
            # Get the events
            events, event_id = mne.events_from_annotations(
                raw_data[ind_i], event_id="auto", verbose=False
            )
            to_remove = []
            for idx in range(len(raw_data[ind_i].annotations.description)):
                if ("boundary" in raw_data[ind_i].annotations.description[idx]) or (
                    "BURST" in raw_data[ind_i].annotations.description[idx]
                ):
                    to_remove.append(idx)

            to_remove = np.array(to_remove)
            if len(to_remove) > 0:
                raw_data[ind_i].annotations.delete(to_remove)
            temp_event, temp_event_id = mne.events_from_annotations(
                raw_data[ind_i], event_id="auto", verbose=False
            )
            temp_event, temp_event_id = self._changeEventID(temp_event, temp_event_id)
            events_list.append(temp_event)
            events_id_list.append(temp_event_id)
            epochs_list.append(
                mne.Epochs(
                    raw_data[ind_i],
                    temp_event,
                    event_id=temp_event_id,
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

        X_parent = np.zeros(
            (
                data_list.shape[0],
                self.length * data_list.shape[1],
                data_list.shape[2],
                self.n_samples_windows,
            )
        )
        Y_parent = np.zeros((data_list.shape[0], self.length * data_list.shape[1]))
        domains_parent = []
        idx_taken = np.zeros((data_list.shape[0], self.length * data_list.shape[1]))
        for ind_i, i in enumerate(self.participants):
            X_parent[ind_i], Y_parent[ind_i], idx_taken[ind_i] = self.to_window(
                data_list[ind_i],
                labels_code_list[ind_i],
                self.length,
                self.n_samples_windows,
                codes,
                window_size=self.window_size,
                sfreq=self.freqwise,
            )
            domains_parent.append(
                [
                    "Source_sub_{}".format(ind_i + 1),
                ]
                * len(Y_parent[ind_i])
            )

        return X_parent, Y_parent, domains_parent, codes, labels_code_list
