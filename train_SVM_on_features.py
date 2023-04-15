import os
import numpy as np
import pandas as pd
import h5py
import mne
from scipy import stats
import scipy.io


mne.set_log_level('error')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from utils.load import Load
from config.default import cfg



def train_SVM(data, finger1, finger2):
    print(f'Training SVM for {finger1} vs {finger2}')

    X = np.concatenate((data[finger1], data[finger2]), axis=0)
    y = np.concatenate((np.ones(data[finger1].shape[0]), np.zeros(data[finger2].shape[0])), axis=0)


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                        'C': [ 1, ]}]
    grid = GridSearchCV(SVC(), tuned_parameters, cv=StratifiedKFold(n_splits=10), scoring='accuracy')
    grid.fit(X, y)
    acc = grid.best_score_* 100
    print(str(round(acc, 2)))
    return acc

def process_subject(subject_id, tag):
    # Load the dictionary from the HDF5 file
    target_dir = 'features'
    file_path = os.path.join(target_dir, tag + '_' + cfg['subjects'][subject_id] + '.h5')


    data = {}
    with h5py.File(file_path, 'r') as h5file:
        for key in h5file.keys():
            data[key] = np.array(h5file[key])

    result = {}
    # RUN trainings
    for finger1 in data:
        for finger2 in data:
            if finger1 != finger2:
                acc = train_SVM(data, finger1, finger2)
                result[finger1 + '_' + finger2] = acc
            else:
                break
    return result

def main():
    #Params
    tag = 'reproduced'
    results =  pd.DataFrame(columns=['subject', 'finger1', 'finger2', 'accuracy'])

    for subject_id in range(5):
        subject = cfg['subjects'][subject_id]
        print("Processing Subject: ", subject)
        result = process_subject(subject_id, tag)
        for key in result:
            finger1, finger2 = key.split('_')
            results = results._append({'subject': subject, 'finger1': finger1, 'finger2': finger2, 'accuracy': result[key]}, ignore_index=True)

    results.to_csv('results/features_SVM.csv', index=False)


if __name__ == "__main__":
    main()