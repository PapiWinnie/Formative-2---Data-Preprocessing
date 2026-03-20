# Formative 2: Multimodal Data Preprocessing
## User Identity and Product Recommendation System

**Group:** Winston · Miracle · Mahe  
**Institution:** African Leadership University · 2025

---

## Overview

A three-stage pipeline that authenticates a user before making a product recommendation. It checks your face, then your voice, and only if both match does it run the recommendation model. Built using customer social media profiles and transaction data, with facial images and voice recordings from each group member.

---

## Project Structure

```
formative2/
├── data/
│   ├── customer_social_profiles.xlsx
│   └── customer_transactions.xlsx
├── images/
│   ├── member_1/          ← Winston (neutral.jpg, smiling.jpg, surprised.jpg)
│   ├── member_2/          ← Miracle
│   └── member_3/          ← Mahe
├── audio/
│   ├── member_1/          ← Winston (yes_approve.wav, confirm_transaction.wav)
│   ├── member_2/          ← Miracle
│   └── member_3/          ← Mahe
├── features/              ← Generated CSVs and plots
├── models/                ← Saved .pkl models
├── scripts/
│   ├── task1_data_merge.py
│   ├── task2_image_processing.py
│   ├── task3_audio_processing.py
│   ├── task4_model_training.py
│   └── task6_system_simulation.py
├── notebooks/
│   └── formative2_notebook.ipynb
└── run_all.py
```

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn Pillow librosa soundfile joblib openpyxl
```

---

## Running the Pipeline

### Full pipeline (all tasks in order):
```bash
python run_all.py
```

### Individual tasks:
```bash
python scripts/task1_data_merge.py
python scripts/task2_image_processing.py
python scripts/task3_audio_processing.py
python scripts/task4_model_training.py
```

---

## System Simulation

Run after the full pipeline has completed and all models are saved.

```bash
# Authorised transaction — specific member
python scripts/task6_system_simulation.py --member member_1

# All three members in sequence
python scripts/task6_system_simulation.py --all

# Unauthorised access attempt
python scripts/task6_system_simulation.py --unauthorized
```

---

## Demo

To record the demo, run these four commands in order and capture the terminal output:

```bash
ls features/ models/
python scripts/task6_system_simulation.py --member member_1
python scripts/task6_system_simulation.py --member member_2
python scripts/task6_system_simulation.py --unauthorized
```

---

## Deliverables Checklist

- [x] `features/merged_dataset.csv`
- [x] `features/image_features.csv`
- [x] `features/audio_features.csv`
- [x] `features/eda_plots.png`
- [x] `features/image_samples.png`
- [x] `features/image_augmentations.png`
- [x] `features/audio_visualizations.png`
- [x] `features/audio_augmentations.png`
- [x] `models/face_model.pkl`
- [x] `models/voice_model.pkl`
- [x] `models/product_model.pkl`
- [x] `notebooks/formative2_notebook.ipynb`
- [x] `scripts/task6_system_simulation.py`
- [ ] Video demo link
- [ ] GitHub repository link