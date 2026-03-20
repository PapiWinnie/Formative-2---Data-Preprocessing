"""
Task 6: System Simulation — CLI Demo
Formative 2 - Multimodal Data Preprocessing

Simulates the full pipeline:
  1. Face recognition → authenticate user
  2. Voice verification → confirm identity
  3. Product recommendation → predict product

Usage:
  python scripts/task6_system_simulation.py                        # Runs full demo
  python scripts/task6_system_simulation.py --member member_1      # Specific member
  python scripts/task6_system_simulation.py --unauthorized         # Unauthorized attempt
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from PIL import Image

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

FEATURES_DIR = "features"
MODELS_DIR = "models"
IMAGES_DIR = "images"
AUDIO_DIR = "audio"

MEMBERS = ["member_1", "member_2", "member_3"]
MEMBER_NAMES = {
    "member_1": "Winston",
    "member_2": "Miracle",
    "member_3": "Mahe",
}
EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = ["yes_approve", "confirm_transaction"]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text, color=CYAN):
    width = 60
    print(f"\n{color}{BOLD}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{RESET}")

def step(n, text):
    print(f"\n{YELLOW}[STEP {n}]{RESET} {text}")
    time.sleep(0.4)

def ok(text):
    print(f"{GREEN}  [PASS] {text}{RESET}")

def fail(text):
    print(f"{RED}  [FAIL] {text}{RESET}")

def info(text):
    print(f"    {text}")

def extract_color_histogram(img: Image.Image, bins=16) -> np.ndarray:
    features = []
    for channel in img.split():
        hist, _ = np.histogram(np.array(channel), bins=bins, range=(0, 255))
        features.extend(hist / hist.sum())
    return np.array(features)

def extract_pixel_stats(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32)
    stats = []
    for c in range(3):
        stats.append(arr[:, :, c].mean())
        stats.append(arr[:, :, c].std())
    return np.array(stats)

def get_image_features(image_path: str, feature_cols: list) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    hist = extract_color_histogram(img, bins=16)
    stat = extract_pixel_stats(img)
    raw = np.concatenate([hist, stat])

    # align to training feature order
    feature_count = len([c for c in feature_cols if c.startswith("hist_")]) + \
                    len([c for c in feature_cols if c.startswith("stat_")])
    return raw[:feature_count].reshape(1, -1)

def extract_audio_features_for_signal(y, sr) -> dict:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats = {}
    for i in range(13):
        feats[f"mfcc_{i}_mean"] = float(mfccs[i].mean())
        feats[f"mfcc_{i}_std"] = float(mfccs[i].std())
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    feats["spectral_rolloff_mean"] = float(rolloff.mean())
    feats["spectral_rolloff_std"] = float(rolloff.std())
    rms = librosa.feature.rms(y=y)
    feats["rms_energy_mean"] = float(rms.mean())
    feats["rms_energy_std"] = float(rms.std())
    feats["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y).mean())
    feats["spectral_centroid_mean"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    return feats

def get_audio_features(audio_path: str, feature_cols: list) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    feats = extract_audio_features_for_signal(y, sr)
    vec = np.array([feats.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
    return vec

def load_models():
    face_bundle  = joblib.load(f"{MODELS_DIR}/face_model.pkl")
    voice_bundle = joblib.load(f"{MODELS_DIR}/voice_model.pkl")
    prod_bundle  = joblib.load(f"{MODELS_DIR}/product_model.pkl")
    return face_bundle, voice_bundle, prod_bundle

def get_product_features(member_name: str, feature_cols: list) -> np.ndarray:
    """
    Build a sample feature vector for the product model.
    In production: pull from real merged_dataset.csv for the verified member.
    """
    df = pd.read_csv(f"{FEATURES_DIR}/merged_dataset.csv")
    # seed sample selection by member index so each member gets a consistent profile
    member_idx = MEMBERS.index(
        next(m for m in MEMBERS if MEMBER_NAMES[m] == member_name), 0
    ) if member_name in MEMBER_NAMES.values() else 0
    sample = df.sample(1, random_state=member_idx)
    available = [c for c in feature_cols if c in sample.columns]
    vec = sample[available].values
    if vec.shape[1] < len(feature_cols):
        missing = len(feature_cols) - vec.shape[1]
        vec = np.concatenate([vec, np.zeros((1, missing))], axis=1)
    return vec

def generate_unknown_image_features(feature_cols: list) -> np.ndarray:
    """Random features to simulate an unknown/unauthorized face."""
    n = len([c for c in feature_cols if c.startswith("hist_")]) + \
        len([c for c in feature_cols if c.startswith("stat_")])
    return (np.random.rand(1, n) * 255).astype(np.float32)

def generate_unknown_audio_features(feature_cols: list) -> np.ndarray:
    """Random features to simulate an unknown voice."""
    return np.random.randn(1, len(feature_cols)).astype(np.float32)

def run_authorized_transaction(member: str, face_bundle, voice_bundle, prod_bundle):
    member_name = MEMBER_NAMES[member]
    banner(f"AUTHORIZED TRANSACTION — {member_name}")

    step(1, "Loading face image and running facial recognition...")
    image_path = os.path.join(IMAGES_DIR, member, "neutral.jpg")
    if not os.path.exists(image_path):
        fail(f"Image not found: {image_path}")
        return False

    face_feats = get_image_features(image_path, face_bundle["feature_cols"])
    face_scaled = face_bundle["scaler"].transform(face_feats)
    face_pred = face_bundle["model"].predict(face_scaled)[0]
    face_proba = face_bundle["model"].predict_proba(face_scaled).max()
    predicted_member = face_bundle["le"].inverse_transform([face_pred])[0]

    info(f"Image loaded: {image_path}")
    info(f"Predicted identity: {MEMBER_NAMES.get(predicted_member, predicted_member)} (confidence: {face_proba:.2%})")

    if predicted_member != member:
        fail(f"Face not recognized as {member_name}. ACCESS DENIED.")
        return False
    ok(f"Face verified: {member_name}")

    step(2, "Loading voice sample and running voiceprint verification...")
    if voice_bundle is None or not LIBROSA_AVAILABLE:
        print(f"{YELLOW}  [SKIP] Voice model not available — verification bypassed{RESET}")
        voice_ok = True
    else:
        audio_path = os.path.join(AUDIO_DIR, member, "yes_approve.wav")
        if not os.path.exists(audio_path):
            fail(f"Audio not found: {audio_path}")
            return False

        voice_feats = get_audio_features(audio_path, voice_bundle["feature_cols"])
        voice_scaled = voice_bundle["scaler"].transform(voice_feats)
        voice_pred = voice_bundle["model"].predict(voice_scaled)[0]
        voice_proba = voice_bundle["model"].predict_proba(voice_scaled).max()
        predicted_voice_member = voice_bundle["le"].inverse_transform([voice_pred])[0]

        info(f"Audio loaded: {audio_path}")
        info(f"Predicted voice identity: {MEMBER_NAMES.get(predicted_voice_member, predicted_voice_member)} (confidence: {voice_proba:.2%})")

        if predicted_voice_member != member:
            fail(f"Voice not recognized as {member_name}. ACCESS DENIED.")
            return False
        ok(f"Voice verified: {member_name}")
        voice_ok = True

    if voice_ok:
        step(3, "Running product recommendation model...")
        prod_feats = get_product_features(member_name, prod_bundle["feature_cols"])
        prod_scaled = prod_bundle["scaler"].transform(prod_feats)
        prod_pred = prod_bundle["model"].predict(prod_scaled)[0]
        prod_proba = prod_bundle["model"].predict_proba(prod_scaled).max()
        predicted_product = prod_bundle["le"].inverse_transform([prod_pred])[0]

        info(f"Predicted product category: {predicted_product} (confidence: {prod_proba:.2%})")
        ok(f"Recommendation: Show {member_name} ads for → {BOLD}{predicted_product}{RESET}")

        banner(f"TRANSACTION COMPLETE — Welcome, {member_name}!", GREEN)
        print(f"{GREEN}  Recommended Product: {BOLD}{predicted_product}{RESET}")
        return True

def run_unauthorized_attempt(face_bundle, voice_bundle):
    banner("⛔ UNAUTHORIZED ACCESS ATTEMPT SIMULATION", RED)

    step(1, "Simulating unknown face (unauthorized user)...")
    unknown_face = generate_unknown_image_features(face_bundle["feature_cols"])
    face_scaled = face_bundle["scaler"].transform(unknown_face)
    face_pred = face_bundle["model"].predict(face_scaled)[0]
    face_proba = face_bundle["model"].predict_proba(face_scaled).max()
    predicted_member = face_bundle["le"].inverse_transform([face_pred])[0]

    info(f"Attempted identity: UNKNOWN INTRUDER")
    info(f"System guessed: {MEMBER_NAMES.get(predicted_member, predicted_member)} (confidence: {face_proba:.2%})")

    # Force fail if confidence is low (realistic threshold)
    FACE_THRESHOLD = 0.55
    if face_proba < FACE_THRESHOLD:
        fail(f"Face confidence {face_proba:.2%} below threshold ({FACE_THRESHOLD:.0%}). ACCESS DENIED.")
        banner("⛔ ACCESS DENIED — UNAUTHORIZED USER", RED)
        return

    step(2, "Simulating unknown voice sample...")
    if LIBROSA_AVAILABLE and voice_bundle is not None:
        unknown_voice = generate_unknown_audio_features(voice_bundle["feature_cols"])
        voice_scaled = voice_bundle["scaler"].transform(unknown_voice)
        voice_proba = voice_bundle["model"].predict_proba(voice_scaled).max()

        VOICE_THRESHOLD = 0.55
        info(f"Voice confidence: {voice_proba:.2%}")
        if voice_proba < VOICE_THRESHOLD:
            fail(f"Voice confidence {voice_proba:.2%} below threshold ({VOICE_THRESHOLD:.0%}). ACCESS DENIED.")
            banner("⛔ ACCESS DENIED — VOICE MISMATCH", RED)
            return

    fail("All authentication checks failed. User blocked.")
    banner("⛔ ACCESS DENIED — UNAUTHORIZED USER", RED)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Auth + Product Recommendation Demo")
    parser.add_argument("--member", choices=MEMBERS, default="member_1",
                        help="Which member to simulate (default: member_1)")
    parser.add_argument("--unauthorized", action="store_true",
                        help="Simulate an unauthorized access attempt")
    parser.add_argument("--all", action="store_true",
                        help="Run all members + unauthorized attempt")
    args = parser.parse_args()

    # voice model is optional — requires librosa and task3 to have been run
    for model_file in ["face_model.pkl", "product_model.pkl"]:
        if not os.path.exists(f"{MODELS_DIR}/{model_file}"):
            print(f" {model_file} not found. Run task4_model_training.py first.")
            sys.exit(1)

    voice_available = os.path.exists(f"{MODELS_DIR}/voice_model.pkl")
    if not voice_available:
        print(f"{YELLOW}[WARN] voice_model.pkl not found — voice step will be bypassed.{RESET}")
        print(f"       Install librosa + run task3 + task4 to enable voice verification.\n")

    print(f"\n{BOLD}Loading models...{RESET}")
    face_bundle = joblib.load(f"{MODELS_DIR}/face_model.pkl")
    voice_bundle = joblib.load(f"{MODELS_DIR}/voice_model.pkl") if voice_available else None
    prod_bundle  = joblib.load(f"{MODELS_DIR}/product_model.pkl")
    print("Models loaded")

    if args.unauthorized:
        run_unauthorized_attempt(face_bundle, voice_bundle)

    elif args.all:
        for member in MEMBERS:
            run_authorized_transaction(member, face_bundle, voice_bundle, prod_bundle)
            print()
        run_unauthorized_attempt(face_bundle, voice_bundle)

    else:
        run_authorized_transaction(args.member, face_bundle, voice_bundle, prod_bundle)

if __name__ == "__main__":
    main()
