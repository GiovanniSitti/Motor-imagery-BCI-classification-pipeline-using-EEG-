from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mne.filter import filter_data
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline


# Configuration

DATA_DIR = Path(r"C:\Users\sitti\AppData\Local\Temp")
TRAIN_MAT = DATA_DIR / "data_set_IVb_al_train.mat"
TEST_MAT = DATA_DIR / "data_set_IVb_al_test.mat"
TRUE_LABELS_MAT = DATA_DIR / "true_labels.mat"

OUTPUT_DIR = Path(__file__).resolve().parent / "motor_imagery_csp_toolbox_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_START_SEC = 0.5
EPOCH_END_SEC = 3.5
BANDPASS_LOW_HZ = 7.0
BANDPASS_HIGH_HZ = 30.0
CSP_COMPONENTS = 6
ONLINE_WINDOW_SAMPLES = 200
PREDICTION_BATCH_SIZE = 512


# Data loading

def load_bci_mat_files():
    train = loadmat(TRAIN_MAT, squeeze_me=True, struct_as_record=False)
    test = loadmat(TEST_MAT, squeeze_me=True, struct_as_record=False)
    labels = loadmat(TRUE_LABELS_MAT, squeeze_me=True, struct_as_record=False)
    return train, test, labels


def get_sampling_rate(nfo) -> float:
    return float(np.asarray(nfo.fs).squeeze())


def get_training_markers(mrk):
    positions = np.asarray(mrk.pos).astype(int).ravel()
    labels = np.asarray(mrk.y).astype(int).ravel()
    return positions, labels


# Preprocessing

def bandpass_eeg(eeg_samples_by_channels: np.ndarray, fs: float) -> np.ndarray:
    eeg_channels_by_samples = eeg_samples_by_channels.T.astype(float)
    filtered = filter_data(
        eeg_channels_by_samples,
        sfreq=fs,
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        method="fir",
        verbose=False,
    )
    return filtered.T


def extract_training_epochs(filtered_eeg: np.ndarray, marker_positions: np.ndarray, labels: np.ndarray, fs: float):
    start = int(round(EPOCH_START_SEC * fs))
    stop = int(round(EPOCH_END_SEC * fs))
    offsets = np.arange(start, stop + 1)

    epochs = []
    epoch_labels = []

    for pos, label in zip(marker_positions, labels):
        indices = pos - 1 + offsets
        if indices[0] >= 0 and indices[-1] < filtered_eeg.shape[0]:
            epoch = filtered_eeg[indices, :].T
            epochs.append(epoch)
            epoch_labels.append(label)

    return np.asarray(epochs), np.asarray(epoch_labels, dtype=int)


# Model training

def build_csp_lda_pipeline() -> Pipeline:
    csp = CSP(
        n_components=CSP_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
        transform_into="average_power",
    )

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    )

    return Pipeline(
        steps=[
            ("csp", csp),
            ("lda", lda),
        ]
    )


def train_model(train_epochs: np.ndarray, train_labels: np.ndarray) -> Pipeline:
    model = build_csp_lda_pipeline()
    model.fit(train_epochs, train_labels)
    return model


# Pseudo-online testing

def iter_online_windows(filtered_eeg: np.ndarray, valid_indices: np.ndarray, window_samples: int):
    for index in valid_indices:
        if index >= window_samples - 1:
            segment = filtered_eeg[index - window_samples + 1 : index + 1, :].T
            yield index, segment


def predict_pseudo_online(model: Pipeline, filtered_test_eeg: np.ndarray, true_y: np.ndarray):
    valid_indices = np.where(np.isin(true_y, [-1, 1]))[0]

    used_indices = []
    predictions = []
    batch = []
    batch_indices = []

    for index, segment in iter_online_windows(filtered_test_eeg, valid_indices, ONLINE_WINDOW_SAMPLES):
        batch.append(segment)
        batch_indices.append(index)

        if len(batch) == PREDICTION_BATCH_SIZE:
            batch_predictions = model.predict(np.asarray(batch))
            predictions.extend(batch_predictions.tolist())
            used_indices.extend(batch_indices)
            batch = []
            batch_indices = []

    if batch:
        batch_predictions = model.predict(np.asarray(batch))
        predictions.extend(batch_predictions.tolist())
        used_indices.extend(batch_indices)

    used_indices = np.asarray(used_indices, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    targets = true_y[used_indices].astype(int)

    return targets, predictions, used_indices


# Evaluation

def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    cm = confusion_matrix(targets, predictions, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(targets, predictions)
    sensitivity = recall_score(targets, predictions, pos_label=1)
    specificity = recall_score(targets, predictions, pos_label=-1)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    misclassification = 1.0 - accuracy

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy_percent": 100.0 * accuracy,
        "sensitivity_percent": 100.0 * sensitivity,
        "specificity_percent": 100.0 * specificity,
        "balanced_accuracy_percent": 100.0 * balanced_accuracy,
        "misclassification_percent": 100.0 * misclassification,
    }


def save_outputs(metrics: dict, train_epochs: np.ndarray, train_labels: np.ndarray, used_indices: np.ndarray):
    summary = {
        "dataset": "BCI Competition III, data set IVb, subject al",
        "classifier": "MNE CSP + scikit-learn shrinkage LDA",
        "training_epochs": train_epochs.shape[0],
        "channels": train_epochs.shape[1],
        "samples_per_epoch": train_epochs.shape[2],
        "class_minus_1_training_epochs": int(np.sum(train_labels == -1)),
        "class_plus_1_training_epochs": int(np.sum(train_labels == 1)),
        "pseudo_online_test_predictions": len(used_indices),
        **metrics,
    }

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    ).to_csv(OUTPUT_DIR / "csp_lda_toolbox_results.csv", index=False)


# Main

def main():
    train, test, labels = load_bci_mat_files()

    fs = get_sampling_rate(train["nfo"])
    train_cnt = np.asarray(train["cnt"], dtype=float)
    test_cnt = np.asarray(test["cnt"], dtype=float)
    true_y = np.asarray(labels["true_y"], dtype=float).ravel()

    marker_positions, marker_labels = get_training_markers(train["mrk"])

    train_filtered = bandpass_eeg(train_cnt, fs)
    test_filtered = bandpass_eeg(test_cnt, fs)

    train_epochs, train_labels = extract_training_epochs(
        train_filtered,
        marker_positions,
        marker_labels,
        fs,
    )

    model = train_model(train_epochs, train_labels)

    targets, predictions, used_indices = predict_pseudo_online(
        model,
        test_filtered,
        true_y,
    )

    metrics = compute_metrics(targets, predictions)
    save_outputs(metrics, train_epochs, train_labels, used_indices)

    print("Motor-imagery CSP toolbox pipeline completed.")
    print(f"Training epochs: {train_epochs.shape[0]}")
    print(f"CSP components: {CSP_COMPONENTS}")
    print(f"Accuracy: {metrics['accuracy_percent']:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity_percent']:.2f}%")
    print(f"Specificity: {metrics['specificity_percent']:.2f}%")
    print(f"Misclassification: {metrics['misclassification_percent']:.2f}%")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

