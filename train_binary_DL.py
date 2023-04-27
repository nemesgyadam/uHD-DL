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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch

import torch.nn as nn

import torch.optim as optim

import optuna


from utils.load import Load
from config.default import cfg


# Set seed for NumPy

np.random.seed(42)


# Set seed for Python's built-in random number generator

random.seed(42)


# Set seed for PyTorch

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


def objective(trial, train_X, test_X, train_y, test_y):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 100, 2000)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    activation_name = trial.suggest_categorical(
        "activation", ["relu", "elu", "leaky_relu"]
    )
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    if activation_name == "relu":
        activation = nn.ReLU()
    elif activation_name == "elu":
        activation = nn.ELU()
    elif activation_name == "leaky_relu":
        activation = nn.LeakyReLU()

    if optimizer == "SGD":
        optimizer = optim.SGD
    elif optimizer == "Adam":
        optimizer = optim.Adam

    model = SingleLayerMLP(train_X.shape[1], hidden_size, 2, activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    return train(
        train_X,
        train_y,
        test_X,
        test_y,
        model,
        criterion,
        optimizer,
        num_epochs=num_epochs,
    )


def train_MLP(data, finger1, finger2, n_trials):
    # print(f"Training SVM for {finger1} vs {finger2}")

    X = np.concatenate((data[finger1], data[finger2]), axis=0)
    y = np.concatenate(
        (np.ones(data[finger1].shape[0]), np.zeros(data[finger2].shape[0])), axis=0
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, train_X, test_X, train_y, test_y),
        n_trials=n_trials,
    )

    best_trial = study.best_trial

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
