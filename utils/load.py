import os
import scipy
import mne

class Load():

    def __init__(self, cfg):

        self.data_root = cfg['data_root']
        self.subjects = cfg['subjects']
        self.dominant_hand = cfg['dominant_hand']
        self.mapping = cfg['mapping']
        self.not_ROI_channels = cfg['not_ROI_channels']
        self.bad_channels = cfg['bad_channels']


        self.left_handed_montage = self.get_montage('left')
        self.right_handed_montage = self.get_montage('right')



    def load_run(self, subject_id, run):
        subject = self.subjects[subject_id]
        mat = scipy.io.loadmat(os.path.join(self.data_root, 'rawdata', subject, run))
        data = mat['y'][1:]  # remove timestamp
        ch_names = [f'c{i}' for i in range(1, 257)] + ['STIM']
        info = mne.create_info(ch_names=ch_names, sfreq=mat['SR\x00'][0][0])

        raw = mne.io.RawArray(data, info)
        ch_types = {ch: 'eeg' if ch != 'STIM' else 'stim' for ch in ch_names}
        raw.set_channel_types(ch_types)

        events = mne.find_events(raw, stim_channel='STIM')
        annot_from_events = mne.annotations_from_events(events, event_desc=self.mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)
        raw.drop_channels(['STIM'])

        montage_positions = self.left_handed_montage if self.dominant_hand[subject_id] == 'left' else self.right_handed_montage
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, montage_positions)), coord_frame='head')
        raw.set_montage(montage)

        return raw

    def load_subject(self, subject_id):
        runs = []
        run_files = os.listdir(os.path.join(self.data_root, 'rawdata', self.subjects[subject_id]))
        for file in run_files:
            runs.append(self.load_run(subject_id, file))
        return runs

    def load_subjects(self, subject_ids):
        subjects = []
        for subject_id in subject_ids:
            subjects.append(self.load_subject(subject_id))
        return subjects

    def get_montage(self, hemishpere):
        mat = scipy.io.loadmat(os.path.join(self.data_root, 'montage', f'montage_256_{hemishpere}_hemisphere.mat'))
        return mat['pos_256']
