from pathlib import Path

import mne
import numpy as np


CLASS_MAP = {
    "769": 0,  # Left Hand
    "770": 1,  # Right Hand
    "771": 2,  # Feet
    "772": 3,  # Tongue
}

CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Feet",
    3: "Tongue",
}


def extract_motor_imagery_epochs(raw, tmin=1.0, tmax=4.0):
    """
    Extract motor-imagery epochs from BCI Competition IV Dataset 2a.

    Returns
    -------
    X : ndarray
        Shape: (n_trials, n_channels, n_samples)

    y : ndarray
        Shape: (n_trials,)

    epochs : mne.Epochs
        MNE Epochs object.
    """

    # Convert annotations to MNE events
    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    # Find the numeric event IDs corresponding to 769-772
    selected_event_ids = {}

    for code in CLASS_MAP:
        if code in event_id:
            selected_event_ids[code] = event_id[code]

    if len(selected_event_ids) != 4:
        raise RuntimeError(
            f"Expected event codes 769, 770, 771, 772. "
            f"Found: {list(selected_event_ids.keys())}"
        )

    # Create event_id with STRING keys.
    # This is required by MNE Epochs.
    mne_event_id = {
        code: numeric_id
        for code, numeric_id in selected_event_ids.items()
    }

    # Select only motor-imagery events
    selected_events = events[
        np.isin(
            events[:, 2],
            list(selected_event_ids.values())
        )
    ]

    if len(selected_events) == 0:
        raise RuntimeError(
            "No motor-imagery events were found."
        )

    # Select EEG channels only.
    # EOG channels are excluded.
    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=False,
        exclude="bads",
    )

    # BCI Competition IV Dataset 2a has 22 EEG + 3 EOG channels.
    # MNE may initially interpret the EOG channels as EEG because
    # of the GDF channel metadata, so explicitly remove them.
    eog_names = {
        "EOG-left",
        "EOG-central",
        "EOG-right",
    }

    eeg_picks = [
        pick
        for pick in eeg_picks
        if raw.ch_names[pick] not in eog_names
    ]

    # Create epochs
    epochs = mne.Epochs(
        raw,
        selected_events,
        event_id=mne_event_id,
        tmin=tmin,
        tmax=tmax,
        picks=eeg_picks,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    # EEG data
    X = epochs.get_data()

    # Convert event names to integer labels
    y = np.array(
        [
            CLASS_MAP[event_name]
            for event_name in epochs.event_id
        ]
    )

    # The above gives one label per event type, not per trial.
    # Therefore construct labels from the actual epoch events.
    reverse_event_map = {
        numeric_id: CLASS_MAP[code]
        for code, numeric_id in selected_event_ids.items()
    }

    y = np.array(
        [
            reverse_event_map[event[2]]
            for event in epochs.events
        ],
        dtype=int,
    )

    return X, y, epochs


def print_epoch_audit(X, y, epochs):
    """Print validation information for extracted epochs."""

    print("\n" + "=" * 60)
    print("EPOCH EXTRACTION AUDIT")
    print("=" * 60)

    print(f"X shape           : {X.shape}")
    print(f"y shape           : {y.shape}")
    print(f"Number of trials  : {len(X)}")
    print(f"EEG channels      : {X.shape[1]}")
    print(f"Samples per trial : {X.shape[2]}")

    print("\nClass distribution:")

    for label, name in CLASS_NAMES.items():
        count = np.sum(y == label)
        print(f"{name:12s}: {count}")

    print("\nEpoch time range:")
    print(f"Start: {epochs.tmin:.3f} s")
    print(f"End  : {epochs.tmax:.3f} s")

    print("=" * 60)


if __name__ == "__main__":

    DATA_PATH = Path("data/BCI_2a/A01T.gdf")

    print(f"Loading: {DATA_PATH}")

    raw = mne.io.read_raw_gdf(
        DATA_PATH,
        preload=True,
        verbose=False,
    )

    X, y, epochs = extract_motor_imagery_epochs(
    raw, tmin=1.0, tmax=4.0
)
    print_epoch_audit(X, y, epochs)