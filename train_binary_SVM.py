import os
import numpy as np
import pandas as pd
import h5py
import mne
from scipy import stats
import scipy.io
import argparse


mne.set_log_level("error")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from utils.load import Load
from config.default import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Run SVM on all subject")
    parser.add_argument("tag", help="File from features folder (without _S[i])")
    return parser.parse_args()


def train_SVM(data, finger1, finger2):
    #print(f"Training SVM for {finger1} vs {finger2}")

    X = np.concatenate((data[finger1], data[finger2]), axis=0)
    y = np.concatenate(
        (np.ones(data[finger1].shape[0]), np.zeros(data[finger2].shape[0])), axis=0
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tuned_parameters = [{'kernel': ['linear','rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                         'C': [ 1,10,100, ]}]
   
    grid = GridSearchCV(
        SVC(), tuned_parameters, cv=StratifiedKFold(n_splits=10), scoring="accuracy"
    )
    grid.fit(X, y)
    acc = grid.best_score_ * 100
    #print(str(round(acc, 2)))
    return acc


def process_subject(subject_id, tag):
    # Load the dictionary from the HDF5 file
    target_dir = "features"
    file_path = os.path.join(
        target_dir, tag + "_" + cfg["subjects"][subject_id] + ".h5"
    )

    data = {}
    with h5py.File(file_path, "r") as h5file:
        for key in h5file.keys():
            data[key] = np.array(h5file[key])
            data[key] = data[key].reshape(data[key].shape[0], -1)

    # RUN trainings
    result = {}
    for finger1 in data:
        for finger2 in data:
            if finger1 != finger2:
                acc = train_SVM(data, finger1, finger2)
                result[finger1 + "_" + finger2] = acc
            else:  # Don't compare same fingers twice
                break
    return result


def main():
    args = parse_args()

    columns = []
    results_np = np.zeros((5, 10))
    for subject_id in range(5):
        subject = cfg["subjects"][subject_id]
        print("Processing Subject: ", subject)
        result = process_subject(subject_id, args.tag)
        columns = result.keys()
        for i, key in enumerate(result):
            fingers = key
            accuracy = result[key]
            results_np[subject_id, i] = accuracy
         
    results = pd.DataFrame(results_np, columns=columns)

    results.to_csv(f"results/SVM_{args.tag}_binary.csv", index=False)


if __name__ == "__main__":
    main()
