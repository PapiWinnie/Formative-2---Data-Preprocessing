"""
Task 3: Audio Data Collection and Processing
Formative 2 - Multimodal Data Preprocessing

HOW TO ADD YOUR OWN RECORDINGS:
  1. Record audio files and save as WAV (16kHz, mono recommended):
       audio/member_1/yes_approve.wav
       audio/member_1/confirm_transaction.wav
  2. For members 2 and 3, place in audio/member_2/ and audio/member_3/
  3. Re-run — the script detects real files and skips synthetic generation.

RECORDING TIPS:
  - Use Audacity (free) or your phone's voice memo app
  - Export as WAV, 16000 Hz, mono
  - Say clearly: "Yes, approve" and "Confirm transaction"
  - Keep to ~2-3 seconds per clip

Install dependencies: pip install librosa soundfile scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import librosa.display
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARN] librosa not installed. Run: pip install librosa soundfile")

AUDIO_DIR = "audio"
FEATURES_DIR = "features"
PHRASES = ["yes_approve", "confirm_transaction"]
MEMBERS = ["member_1", "member_2", "member_3"]
MEMBER_NAMES = {
    "member_1": "Winston",
    "member_2": "Miracle",
    "member_3": "Mahe",
}
SAMPLE_RATE = 16000
DURATION = 2  # seconds

def generate_synthetic_audio(phrase: str, member_idx: int, sr=SAMPLE_RATE, duration=DURATION) -> np.ndarray:
    """Sine + harmonics + noise waveform; each member uses a different base frequency."""
    t = np.linspace(0, duration, sr * duration)
    base_freq = 120 + (member_idx * 40)  # 160, 200, 240 Hz per member
    phrase_offset = 10 if phrase == "confirm_transaction" else 0

    signal = (
        0.5 * np.sin(2 * np.pi * (base_freq + phrase_offset) * t)
        + 0.25 * np.sin(2 * np.pi * (base_freq * 2 + phrase_offset) * t)
        + 0.1 * np.sin(2 * np.pi * (base_freq * 3) * t)
        + 0.05 * np.random.randn(len(t))
    )
    # Fade in/out
    fade = np.ones(len(t))
    fade_len = sr // 8
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)
    return (signal * fade).astype(np.float32)

def ensure_audio_exists():
    if not LIBROSA_AVAILABLE:
        print("[SKIP] librosa not available, skipping audio generation")
        return
    for i, member in enumerate(MEMBERS, 1):
        member_dir = os.path.join(AUDIO_DIR, member)
        os.makedirs(member_dir, exist_ok=True)
        for phrase in PHRASES:
            path = os.path.join(member_dir, f"{phrase}.wav")
            if not os.path.exists(path):
                audio = generate_synthetic_audio(phrase, i)
                sf.write(path, audio, SAMPLE_RATE)
                print(f"  [SYNTHETIC] Generated placeholder: {path}")

def load_member_audio() -> dict:
    audio_data = {}
    for member in MEMBERS:
        audio_data[member] = {}
        for phrase in PHRASES:
            path = os.path.join(AUDIO_DIR, member, f"{phrase}.wav")
            if os.path.exists(path):
                y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                audio_data[member][phrase] = (y, sr)
            else:
                print(f"  [WARN] Missing: {path}")
    return audio_data

def display_audio_visualizations(audio_data: dict):
    n_members = len(MEMBERS)
    n_phrases = len(PHRASES)
    fig, axes = plt.subplots(n_members * 2, n_phrases, figsize=(14, 4 * n_members))
    fig.suptitle("Audio Samples — Waveforms & Spectrograms", fontsize=13, fontweight="bold")

    for r, member in enumerate(MEMBERS):
        for c, phrase in enumerate(PHRASES):
            if phrase not in audio_data[member]:
                continue
            y, sr = audio_data[member][phrase]
            label = f"{MEMBER_NAMES[member]}\n{phrase.replace('_', ' ').title()}"

            ax_wave = axes[r * 2, c]
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="steelblue")
            ax_wave.set_title(f"Waveform: {label}", fontsize=8)
            ax_wave.set_xlabel("Time (s)")
            ax_wave.set_ylabel("Amplitude")

            ax_spec = axes[r * 2 + 1, c]
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, ax=ax_spec, x_axis="time", y_axis="mel", cmap="magma")
            ax_spec.set_title(f"Mel Spectrogram: {label}", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FEATURES_DIR}/audio_visualizations.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[AUDIO] Visualizations saved → features/audio_visualizations.png")

def augment_audio(y: np.ndarray, sr: int) -> list:
    """Returns list of (augmented_signal, augmentation_name): pitch shift, time stretch, noise."""
    augmented = []

    try:
        pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented.append((pitched, "pitch_shift"))
    except Exception:
        augmented.append((y.copy(), "pitch_shift_fallback"))

    try:
        stretched = librosa.effects.time_stretch(y, rate=0.85)
        # pad or trim to original length
        if len(stretched) < len(y):
            stretched = np.pad(stretched, (0, len(y) - len(stretched)))
        else:
            stretched = stretched[:len(y)]
        augmented.append((stretched, "time_stretch"))
    except Exception:
        augmented.append((y.copy(), "time_stretch_fallback"))

    noise = 0.008 * np.random.randn(len(y))
    noisy = (y + noise).astype(np.float32)
    augmented.append((noisy, "background_noise"))

    return augmented

def extract_audio_features_for_signal(y: np.ndarray, sr: int) -> dict:
    """Extract MFCCs, spectral roll-off, and energy from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_features = {}
    for i in range(13):
        mfcc_features[f"mfcc_{i}_mean"] = float(mfccs[i].mean())
        mfcc_features[f"mfcc_{i}_std"] = float(mfccs[i].std())

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mfcc_features["spectral_rolloff_mean"] = float(rolloff.mean())
    mfcc_features["spectral_rolloff_std"] = float(rolloff.std())

    rms = librosa.feature.rms(y=y)
    mfcc_features["rms_energy_mean"] = float(rms.mean())
    mfcc_features["rms_energy_std"] = float(rms.std())

    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc_features["zcr_mean"] = float(zcr.mean())

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mfcc_features["spectral_centroid_mean"] = float(centroid.mean())

    return mfcc_features

def extract_all_audio_features(audio_data: dict) -> pd.DataFrame:
    records = []
    for member in MEMBERS:
        for phrase in PHRASES:
            if phrase not in audio_data[member]:
                continue
            y, sr = audio_data[member][phrase]

            feats = extract_audio_features_for_signal(y, sr)
            records.append({
                "member": member,
                "member_name": MEMBER_NAMES[member],
                "phrase": phrase,
                "augmentation": "original",
                **feats,
            })

            for aug_y, aug_name in augment_audio(y, sr):
                feats = extract_audio_features_for_signal(aug_y, sr)
                records.append({
                    "member": member,
                    "member_name": MEMBER_NAMES[member],
                    "phrase": phrase,
                    "augmentation": aug_name,
                    **feats,
                })

    df = pd.DataFrame(records)
    df.to_csv(f"{FEATURES_DIR}/audio_features.csv", index=False)
    print(f"[AUDIO] audio_features.csv saved → {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def display_augmentation_comparison(audio_data: dict):
    """Show waveform comparison of original vs augmentations for member_1."""
    member = "member_1"
    phrase = "yes_approve"
    if phrase not in audio_data[member]:
        return

    y, sr = audio_data[member][phrase]
    augs = augment_audio(y, sr)
    all_signals = [(y, "Original")] + [(a[0], a[1].replace("_", " ").title()) for a in augs]

    fig, axes = plt.subplots(1, len(all_signals), figsize=(16, 3))
    fig.suptitle(f"Audio Augmentations — {MEMBER_NAMES[member]}: '{phrase.replace('_',' ')}'",
                 fontsize=11, fontweight="bold")

    for ax, (signal, name) in zip(axes, all_signals):
        librosa.display.waveshow(signal, sr=sr, ax=ax, color="coral")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"{FEATURES_DIR}/audio_augmentations.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[AUDIO] Augmentation comparison saved → features/audio_augmentations.png")

def main():
    if not LIBROSA_AVAILABLE:
        print(" librosa is required. Install with: pip install librosa soundfile")
        return None, None

    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("Checking / generating audio")
    ensure_audio_exists()

    print("\nLoading audio")
    audio_data = load_member_audio()

    print("\nDisplaying waveforms and spectrograms")
    display_audio_visualizations(audio_data)

    print("\nDisplaying augmentation comparison")
    display_augmentation_comparison(audio_data)

    print("\nExtracting audio features")
    df = extract_all_audio_features(audio_data)
    print(df[["member", "member_name", "phrase", "augmentation"]].value_counts().sort_index())

    print("\n Task 3 complete.")
    return audio_data, df

if __name__ == "__main__":
    main()
