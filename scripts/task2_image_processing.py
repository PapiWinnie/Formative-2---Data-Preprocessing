"""
Task 2: Image Data Collection and Processing
Formative 2 - Multimodal Data Preprocessing

HOW TO ADD YOUR OWN PHOTOS:
  1. Place your photos in images/member_1/ (rename if needed):
       images/member_1/neutral.jpg
       images/member_1/smiling.jpg
       images/member_1/surprised.jpg
  2. For members 2 and 3, drop photos in images/member_2/ and images/member_3/
  3. Re-run this script — it auto-detects real images vs synthetic placeholders.

NOTE: When no real images are present, synthetic placeholder images are generated
so the pipeline runs end-to-end. Replace them with actual photos before final submission.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

IMAGES_DIR = "images"
FEATURES_DIR = "features"
EXPRESSIONS = ["neutral", "smiling", "surprised"]
MEMBERS = ["member_1", "member_2", "member_3"]
# Update member names here
MEMBER_NAMES = {
    "member_1": "Winston",
    "member_2": "Miracle",
    "member_3": "Mahe",
}

def generate_synthetic_image(expression: str, member_idx: int, size=(128, 128)) -> Image.Image:
    """Generate a distinguishable placeholder image per member/expression."""
    colors = {
        "member_1": [(200, 150, 120), (220, 180, 140), (180, 120, 100)],
        "member_2": [(120, 160, 200), (140, 180, 220), (100, 140, 180)],
        "member_3": [(150, 200, 130), (170, 220, 150), (130, 180, 110)],
    }
    expression_map = {"neutral": 0, "smiling": 1, "surprised": 2}
    base_color = colors[f"member_{member_idx}"][expression_map[expression]]
    arr = np.full((*size, 3), base_color, dtype=np.uint8)
    # Add noise to differentiate
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def ensure_images_exist():
    """Auto-generate synthetic images for any missing member/expression combos."""
    for i, member in enumerate(MEMBERS, 1):
        member_dir = os.path.join(IMAGES_DIR, member)
        os.makedirs(member_dir, exist_ok=True)
        for expr in EXPRESSIONS:
            path = os.path.join(member_dir, f"{expr}.jpg")
            if not os.path.exists(path):
                img = generate_synthetic_image(expr, i)
                img.save(path)
                print(f"  [SYNTHETIC] Generated placeholder: {path}")

def load_member_images() -> dict:
    images = {}
    for member in MEMBERS:
        images[member] = {}
        for expr in EXPRESSIONS:
            path = os.path.join(IMAGES_DIR, member, f"{expr}.jpg")
            if os.path.exists(path):
                images[member][expr] = Image.open(path).convert("RGB").resize((128, 128))
            else:
                print(f"  [WARN] Missing: {path}")
    return images

def display_sample_images(images: dict):
    fig, axes = plt.subplots(len(MEMBERS), len(EXPRESSIONS), figsize=(12, 4 * len(MEMBERS)))
    fig.suptitle("Facial Image Samples — All Members", fontsize=14, fontweight="bold")

    for r, member in enumerate(MEMBERS):
        for c, expr in enumerate(EXPRESSIONS):
            ax = axes[r, c]
            if expr in images[member]:
                ax.imshow(images[member][expr])
            ax.set_title(f"{MEMBER_NAMES[member]}\n{expr.capitalize()}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{FEATURES_DIR}/image_samples.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[IMG] Sample grid saved → features/image_samples.png")

def augment_image(img: Image.Image, label: str) -> list:
    """Returns list of (augmented_image, augmentation_name) tuples: rotation, flip, grayscale."""
    augmented = []

    rotated = img.rotate(15, expand=False, fillcolor=(128, 128, 128))
    augmented.append((rotated, f"{label}_rotated"))

    flipped = ImageOps.mirror(img)
    augmented.append((flipped, f"{label}_flipped"))

    gray = ImageOps.grayscale(img).convert("RGB")  # kept as RGB for consistency
    augmented.append((gray, f"{label}_grayscale"))

    return augmented

def display_augmentations(images: dict):
    """Display augmentations for one sample image per member."""
    aug_names = ["Original", "Rotated (+15°)", "Flipped (H)", "Grayscale"]
    fig, axes = plt.subplots(len(MEMBERS), 4, figsize=(14, 3.5 * len(MEMBERS)))
    fig.suptitle("Image Augmentations (Sample: Neutral Expression)", fontsize=13, fontweight="bold")

    for r, member in enumerate(MEMBERS):
        base_img = images[member]["neutral"]
        augs = augment_image(base_img, "neutral")
        all_imgs = [base_img] + [a[0] for a in augs]

        for c, (img, name) in enumerate(zip(all_imgs, aug_names)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_title(f"{MEMBER_NAMES[member]}\n{name}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{FEATURES_DIR}/image_augmentations.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[IMG] Augmentation grid saved → features/image_augmentations.png")

def extract_color_histogram(img: Image.Image, bins=16) -> np.ndarray:
    """Flatten RGB histogram as image feature."""
    features = []
    for channel in img.split():
        hist, _ = np.histogram(np.array(channel), bins=bins, range=(0, 255))
        features.extend(hist / hist.sum())  # normalise
    return np.array(features)

def extract_pixel_stats(img: Image.Image) -> np.ndarray:
    """Mean and std per RGB channel."""
    arr = np.array(img, dtype=np.float32)
    stats = []
    for c in range(3):
        stats.append(arr[:, :, c].mean())
        stats.append(arr[:, :, c].std())
    return np.array(stats)

def extract_image_features(images: dict) -> pd.DataFrame:
    records = []
    for member in MEMBERS:
        for expr in EXPRESSIONS:
            img = images[member][expr]
            hist_feats = extract_color_histogram(img, bins=16)  # 48 dims
            stat_feats = extract_pixel_stats(img)                # 6 dims

            aug_list = augment_image(img, f"{member}_{expr}")
            for aug_img, aug_name in aug_list:
                aug_hist = extract_color_histogram(aug_img, bins=16)
                aug_stat = extract_pixel_stats(aug_img)
                row = {
                    "member": member,
                    "member_name": MEMBER_NAMES[member],
                    "expression": expr,
                    "augmentation": aug_name.split("_")[-1],
                    **{f"hist_{i}": v for i, v in enumerate(aug_hist)},
                    **{f"stat_{i}": v for i, v in enumerate(aug_stat)},
                }
                records.append(row)

            row = {
                "member": member,
                "member_name": MEMBER_NAMES[member],
                "expression": expr,
                "augmentation": "original",
                **{f"hist_{i}": v for i, v in enumerate(hist_feats)},
                **{f"stat_{i}": v for i, v in enumerate(stat_feats)},
            }
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(f"{FEATURES_DIR}/image_features.csv", index=False)
    print(f"[IMG] image_features.csv saved → {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    print("Checking / generating images")
    ensure_images_exist()

    print("\nLoading images")
    images = load_member_images()

    print("\nDisplaying sample images")
    display_sample_images(images)

    print("\nDisplaying augmentations")
    display_augmentations(images)

    print("\nExtracting image features")
    df = extract_image_features(images)
    print(df[["member", "member_name", "expression", "augmentation"]].value_counts().sort_index())

    print("\n Task 2 complete.")
    return images, df

if __name__ == "__main__":
    main()
