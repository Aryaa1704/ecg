"""Reference-paper experiments for BCI Competition IV Dataset 2a.

Protocol:
- Train on T session; test on E session using official MAT labels.
- Third-order 0.5-30 Hz Butterworth bandpass.
- Binary left-vs-right CSP-LDA: four window widths and 3-6 s baseline.
- Four-class CSP-LDA: six OVO classifiers, four CSP features,
  3-second sliding windows and 3-6 s baseline.
- Sliding-window times are relative to visual cue onset (events 769-772).
  In this dataset, cue onset is 2 seconds after event 768.
"""

from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from mne.decoding import CSP


DATA_DIR = Path("data/BCI_2a")
OUT_DIR = Path("results/paper_reproduction/tables")

SFREQ_EXPECTED = 250
CLASS_EVENT_TO_LABEL = {
    "769": 0,  # left hand
    "770": 1,  # right hand
    "771": 2,  # feet
    "772": 3,  # tongue
}
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]

BINARY_WIDTHS = [2.0, 2.5, 3.0, 3.5]
SLIDE_STEP = 0.1
BASELINE = (3.0, 6.0)
FOUR_CLASS_WIDTH = 3.0
CUE_ONSET_OFFSET_SEC = 2.0


def read_gdf(path):
    """Load a GDF file and return the raw object."""
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], SFREQ_EXPECTED):
        raise ValueError(
            f"{path}: expected {SFREQ_EXPECTED} Hz, got {raw.info['sfreq']}"
        )
    return raw


def annotation_events(raw):
    """Return annotation events with sample indices relative to raw data."""
    events, event_id = mne.events_from_annotations(
        raw, event_id="auto", verbose="ERROR"
    )
    reverse_ids = {int(value): str(key) for key, value in event_id.items()}
    descriptions = [reverse_ids.get(int(code), "") for code in events[:, 2]]
    samples = events[:, 0] - raw.first_samp
    return list(zip(samples.astype(int), descriptions))


def get_trial_starts_and_training_labels(raw, path):
    """Pair event-768 trial starts with their 769-772 class cues."""
    events = annotation_events(raw)
    starts = [sample for sample, desc in events if desc == "768"]
    cues = [
        (sample, CLASS_EVENT_TO_LABEL[desc])
        for sample, desc in events
        if desc in CLASS_EVENT_TO_LABEL
    ]

    if len(starts) != 288:
        raise ValueError(f"{path}: expected 288 trial starts, found {len(starts)}")
    if len(cues) != 288:
        raise ValueError(f"{path}: expected 288 class cues, found {len(cues)}")

    starts = np.asarray(starts, dtype=int)
    cue_samples = np.asarray([item[0] for item in cues], dtype=int)
    labels = np.asarray([item[1] for item in cues], dtype=int)

    # Check that each cue follows its corresponding trial start by 2 seconds.
    delays = cue_samples - starts
    expected_delay = int(round(2.0 * raw.info["sfreq"]))
    if not np.all(delays == expected_delay):
        bad = np.where(delays != expected_delay)[0][:5].tolist()
        raise ValueError(
            f"{path}: class cues are not consistently 2 s after trial starts; "
            f"first mismatched trial indices: {bad}"
        )
    return starts, labels


def load_official_evaluation_labels(path):
    """Load the provided evaluation labels from the subject's MAT file."""
    mat = loadmat(path)
    if "classlabel" not in mat:
        keys = [key for key in mat if not key.startswith("__")]
        raise KeyError(f"{path}: 'classlabel' not found. MAT keys: {keys}")

    labels = np.asarray(mat["classlabel"]).reshape(-1).astype(int)
    if len(labels) != 288:
        raise ValueError(f"{path}: expected 288 labels, found {len(labels)}")
    if not np.isin(labels, [1, 2, 3, 4]).all():
        raise ValueError(f"{path}: expected official labels 1, 2, 3, 4")

    # Official MAT labels are 1-based; Python class labels are 0-based.
    return labels - 1


def filtered_eeg(raw, path):
    """Select 22 EEG channels and apply the paper's Butterworth bandpass."""
    # The GDF reader may label EOG channels as EEG.
    # Explicitly exclude the three named EOG channels.
    eog_names = {"EOG-left", "EOG-central", "EOG-right"}
    eeg_picks = [
        i for i, name in enumerate(raw.ch_names)
        if name not in eog_names
    ]

    if len(eeg_picks) != 22:
        raise ValueError(
            f"{path}: expected 22 EEG channels after excluding EOG; "
            f"found {len(eeg_picks)}: {[raw.ch_names[i] for i in eeg_picks]}"
        )

    data = raw.get_data(picks=eeg_picks)
    sos = butter(
        3,
        [0.5, 30.0],
        btype="bandpass",
        fs=raw.info["sfreq"],
        output="sos",
    )
    # Filter the continuous EEG before extracting windows.
    data = sosfiltfilt(sos, data, axis=-1)
    return data


def extract_windows(data, trial_starts, sfreq, start_sec, width_sec):
    """Extract windows relative to cue onset; event 768 precedes cue by 2 s."""
    cue_offset = int(round(CUE_ONSET_OFFSET_SEC * sfreq))
    start_offset = int(round(start_sec * sfreq))
    length = int(round(width_sec * sfreq))
    epochs = []

    for trial_start in trial_starts:
        begin = int(trial_start) + cue_offset + start_offset
        end = begin + length
        if begin < 0 or end > data.shape[-1]:
            raise ValueError(
                f"Window {start_sec}-{start_sec + width_sec} s exceeds recording"
            )
        epochs.append(data[:, begin:end])

    return np.asarray(epochs, dtype=np.float64)


def make_classifier(experiment):
    """Build CSP-LDA; fit a separate CSP-LDA pipeline for each OVO pair."""
    def make_csp_lda():
        return make_pipeline(
            CSP(
                n_components=4,
                reg=None,
                log=True,
                norm_trace=False,
                component_order="mutual_info",
            ),
            LinearDiscriminantAnalysis(),
        )

    if experiment == "four_class_ovo":
        return OneVsOneClassifier(make_csp_lda())

    return make_csp_lda()


def evaluate_window(
    subject, experiment, width, start_sec, end_sec,
    train_data, train_starts, train_labels,
    test_data, test_starts, test_labels, sfreq,
):
    """Fit on T and evaluate once on E for one window."""
    x_train = extract_windows(train_data, train_starts, sfreq, start_sec, width)
    x_test = extract_windows(test_data, test_starts, sfreq, start_sec, width)

    if experiment == "binary_left_right":
        train_mask = np.isin(train_labels, [0, 1])
        test_mask = np.isin(test_labels, [0, 1])
        x_train, y_train = x_train[train_mask], train_labels[train_mask]
        x_test, y_test = x_test[test_mask], test_labels[test_mask]
    else:
        y_train, y_test = train_labels, test_labels

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{subject}: fewer than two training classes")

    model = make_classifier(experiment)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "subject": subject,
        "experiment": experiment,
        "window_start_cue_sec": round(start_sec, 1),
        "window_end_cue_sec": round(end_sec, 1),
        "window_width_sec": round(width, 1),
        "accuracy": accuracy_score(y_test, predictions),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def windows_for_width(width):
    """Paper's sliding windows: 2.0 to 6.5 s, in 0.1 s steps."""
    latest_start = 6.5 - width
    count = int(round((latest_start - 2.0) / SLIDE_STEP)) + 1
    return [
        round(2.0 + i * SLIDE_STEP, 1)
        for i in range(count)
    ]


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR.resolve()}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for subject_number in range(1, 10):
        subject = f"A{subject_number:02d}"
        train_gdf = DATA_DIR / f"{subject}T.gdf"
        test_gdf = DATA_DIR / f"{subject}E.gdf"
        train_mat = DATA_DIR / f"{subject}T.mat"
        test_mat = DATA_DIR / f"{subject}E.mat"

        for path in [train_gdf, test_gdf, train_mat, test_mat]:
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file missing: {path}")

        print(f"\n[{subject}] Loading T and E sessions...")
        raw_train = read_gdf(train_gdf)
        raw_test = read_gdf(test_gdf)

        train_starts, train_labels = get_trial_starts_and_training_labels(
            raw_train, train_gdf
        )
        test_starts, _ = get_trial_starts_and_training_labels(
            raw_test, test_gdf
        ) if any(
            desc in CLASS_EVENT_TO_LABEL
            for _, desc in annotation_events(raw_test)
        ) else (None, None)

        # Evaluation GDF normally uses event 783 instead of class-specific cues.
        if test_starts is None:
            test_events = annotation_events(raw_test)
            test_starts = np.asarray(
                [sample for sample, desc in test_events if desc == "768"],
                dtype=int,
            )
            if len(test_starts) != 288:
                raise ValueError(
                    f"{test_gdf}: expected 288 trial starts, found {len(test_starts)}"
                )

        test_labels = load_official_evaluation_labels(test_mat)
        if len(test_starts) != len(test_labels):
            raise ValueError(
                f"{subject}: evaluation trial count does not match official labels"
            )

        train_data = filtered_eeg(raw_train, train_gdf)
        test_data = filtered_eeg(raw_test, test_gdf)
        sfreq = raw_train.info["sfreq"]

        # Experiment 1: binary left-hand versus right-hand CSP-LDA.
        binary_windows = []
        for width in BINARY_WIDTHS:
            binary_windows.extend((start, width) for start in windows_for_width(width))
        binary_windows.append((BASELINE[0], BASELINE[1] - BASELINE[0]))

        for start_sec, width in binary_windows:
            rows.append(
                evaluate_window(
                    subject, "binary_left_right", width, start_sec,
                    start_sec + width, train_data, train_starts, train_labels,
                    test_data, test_starts, test_labels, sfreq,
                )
            )

        # Experiment 2: four-class OVO, 3-second sliding windows + baseline.
        four_class_windows = [
            (start, FOUR_CLASS_WIDTH)
            for start in windows_for_width(FOUR_CLASS_WIDTH)
        ]
        four_class_windows.append(
            (BASELINE[0], BASELINE[1] - BASELINE[0])
        )

        for start_sec, width in four_class_windows:
            rows.append(
                evaluate_window(
                    subject, "four_class_ovo", width, start_sec,
                    start_sec + width, train_data, train_starts, train_labels,
                    test_data, test_starts, test_labels, sfreq,
                )
            )

        raw_train.close()
        raw_test.close()
        print(f"[{subject}] Completed both paper experiments.")

    results = pd.DataFrame(rows)
    results_path = OUT_DIR / "subject_window_accuracies.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(
            [
                "experiment",
                "window_start_cue_sec",
                "window_end_cue_sec",
                "window_width_sec",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            subjects=("subject", "nunique"),
        )
    )
    summary_path = OUT_DIR / "mean_accuracy_across_subjects.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFinished.")
    print(f"Per-subject results: {results_path}")
    print(f"Across-subject summary: {summary_path}")
    print("\nHighest mean-accuracy windows by experiment:")
    for experiment, group in summary.groupby("experiment"):
        best = group.loc[group["mean_accuracy"].idxmax()]
        print(
            f"{experiment}: {best['mean_accuracy']:.4f} "
            f"(window {best['window_start_trial_sec']:.1f}-"
            f"{best['window_end_trial_sec']:.1f} s)"
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
