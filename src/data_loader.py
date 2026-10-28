from pathlib import Path

import mne


def load_gdf(file_path: str | Path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw = mne.io.read_raw_gdf(
        file_path,
        preload=True,
        verbose=True,
    )

    return raw


def print_data_audit(raw):
    print("=" * 60)
    print("BCI EEG DATA AUDIT")
    print("=" * 60)

    print(f"Sampling frequency : {raw.info['sfreq']} Hz")
    print(f"Total channels     : {len(raw.ch_names)}")
    print(f"Recording duration : {raw.times[-1]:.2f} seconds")

    print("\nChannel names:")
    for i, channel in enumerate(raw.ch_names, start=1):
        print(f"{i:02d}. {channel}")

    print("\nAnnotations:")
    print(raw.annotations)

    print("=" * 60)
def inspect_event_timing(raw):
    """Inspect timing between trial-start and motor-imagery events."""

    print("\n" + "=" * 60)
    print("EVENT TIMING ANALYSIS")
    print("=" * 60)

    descriptions = raw.annotations.description
    onsets = raw.annotations.onset

    for i in range(len(descriptions) - 1):

        current_event = descriptions[i]
        next_event = descriptions[i + 1]

        if current_event == "768" and next_event in {"769", "770", "771", "772"}:

            delay = onsets[i + 1] - onsets[i]

            print(
                f"768 -> {next_event} : "
                f"{delay:.3f} seconds"
            )

    print("=" * 60)

if __name__ == "__main__":
    DATA_PATH = Path("data/BCI_2a/A01T.gdf")

    print(f"Loading: {DATA_PATH}")

    raw = load_gdf(DATA_PATH)

    print_data_audit(raw)
    inspect_event_timing(raw)
def extract_mi_events(raw):
    """Extract motor-imagery class events and their onset times."""

    descriptions = raw.annotations.description
    onsets = raw.annotations.onset

    class_map = {
        "769": "Left Hand",
        "770": "Right Hand",
        "771": "Feet",
        "772": "Tongue",
    }

    events = []

    for description, onset in zip(descriptions, onsets):
        if description in class_map:
            events.append({
                "onset": onset,
                "code": description,
                "class": class_map[description],
            })

    print("\n" + "=" * 60)
    print("MOTOR IMAGERY EVENTS")
    print("=" * 60)

    print(f"Total MI events: {len(events)}")

    for event in events[:10]:
        print(
            f"{event['onset']:8.3f}s | "
            f"{event['code']} | "
            f"{event['class']}"
        )

    print("\nClass distribution:")

    from collections import Counter

    counts = Counter(event["class"] for event in events)

    for class_name, count in counts.items():
        print(f"{class_name:12s}: {count}")

    print("=" * 60)

    return events
if __name__ == "__main__":
    DATA_PATH = Path("data/BCI_2a/A01T.gdf")

    print(f"Loading: {DATA_PATH}")

    raw = load_gdf(DATA_PATH)

    print_data_audit(raw)
    inspect_event_timing(raw)

    mi_events = extract_mi_events(raw)    