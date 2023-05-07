import os
import numpy as np
import pandas as pd
import random
import h5py
import mne
from scipy import stats
import scipy.io
import argparse


mne.set_log_level("error")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import optuna


from utils.load import Load
from config.default import cfg


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run SVM on all subject")
    parser.add_argument("tag", help="File from features folder (without _S[i])")
    parser.add_argument("--n_trials", default=100, help="Number of trials for optuna")
    return parser.parse_args()


class SingleLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(SingleLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def train(
    X_train, y_train, X_test, y_test, model, criterion, optimizer, num_epochs=100
):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(y_pred, dim=1)

    acc = accuracy_score(y_test.cpu(), y_pred.cpu())
    return acc*100


def train(X_train, y_train, X_test, y_test, model, criterion, optimizer, num_epochs=100):
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
   
   
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)


    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(y_pred, dim=1)

    acc = accuracy_score(y_test.cpu(), y_pred.cpu())
    return acc

def objective(trial, X, y):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 500, 1000)
    hidden_size = trial.suggest_int("hidden_size", 2, 16)
    activation_name = trial.suggest_categorical("activation", ["relu", "elu", "leaky_relu"])
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    if activation_name == "relu":
        activation = nn.ReLU()
    elif activation_name == "elu":
        activation = nn.ELU()
    elif activation_name == "leaky_relu":
        activation = nn.LeakyReLU()

    if optimizer == "SGD":
        optimizer_fnc = optim.SGD
    elif optimizer == "Adam":
        optimizer_fnc = optim.Adam

   
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SingleLayerMLP(X_train.shape[1], hidden_size, 2, activation)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_fnc(model.parameters(), lr=learning_rate)
        acc = train(X_train, y_train, X_test, y_test, model, criterion, optimizer, num_epochs=num_epochs)
        fold_accuracies.append(acc)

    mean_accuracy = np.mean(fold_accuracies)
    return mean_accuracy

def train_MLP(data, finger1, finger2, n_trials = 10, verbose = True):
   
    print(f'Training MLP for {finger1} vs {finger2}')

    X = np.concatenate((data[finger1], data[finger2]), axis=0)
    y = np.concatenate((np.ones(data[finger1].shape[0]), np.zeros(data[finger2].shape[0])), axis=0)

   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    best_trial = study.best_trial

    print(f'Best trial params: {best_trial.params}')
    print(f'Best trial accuracy: {best_trial.value * 100:.2f}%')
    return best_trial.value


def process_subject(subject_id, tag, n_trials):
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
                acc = train_MLP(data, finger1, finger2, n_trials)
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
        result = process_subject(subject_id, args.tag, int(args.n_trials))
        columns = result.keys()
        for i, key in enumerate(result):
            fingers = key
            accuracy = result[key]
            results_np[subject_id, i] = accuracy

    results = pd.DataFrame(results_np, columns=columns)

    results.to_csv(f"results/DL_{args.tag}_binary.csv", index=False)


if __name__ == "__main__":
    main()
