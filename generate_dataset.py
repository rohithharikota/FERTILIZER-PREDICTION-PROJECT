"""
Generate a large synthetic fertilizer recommendation dataset (10,000 rows).
Each fertilizer type has a distinct statistical profile for the input features,
ensuring the ML models can learn meaningful decision boundaries.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'fertilizer_dataset.csv')
TOTAL_ROWS = 10000

# ─── Soil & Crop options ────────────────────────────────────────────────────
SOIL_TYPES = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
CROP_TYPES = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton',
              'Tobacco', 'Millets', 'Pulses', 'Oil Seeds', 'Ground Nuts']

# ─── Fertilizer profiles ────────────────────────────────────────────────────
# Each profile defines (mean, std) for numeric features and preferred soil/crop.
# This ensures each fertilizer has a distinct "signature" in feature space.

PROFILES = {
    'Urea': {
        'temperature': (28, 4),
        'humidity':    (50, 8),
        'moisture':    (40, 7),
        'nitrogen':    (85, 8),     # HIGH nitrogen
        'phosphorus':  (15, 8),
        'potassium':   (12, 7),
        'soil_pref':   ['Clayey', 'Loamy', 'Black'],
        'crop_pref':   ['Rice', 'Maize', 'Sugarcane', 'Cotton', 'Wheat'],
    },
    'DAP': {
        'temperature': (29, 4),
        'humidity':    (60, 8),
        'moisture':    (48, 7),
        'nitrogen':    (16, 7),
        'phosphorus':  (75, 8),     # HIGH phosphorus
        'potassium':   (18, 8),
        'soil_pref':   ['Loamy', 'Black', 'Red'],
        'crop_pref':   ['Wheat', 'Rice', 'Pulses', 'Maize'],
    },
    'MOP': {
        'temperature': (30, 4),
        'humidity':    (44, 7),
        'moisture':    (36, 6),
        'nitrogen':    (8, 5),
        'phosphorus':  (12, 6),
        'potassium':   (72, 8),     # HIGH potassium
        'soil_pref':   ['Black', 'Loamy', 'Red'],
        'crop_pref':   ['Cotton', 'Wheat', 'Maize', 'Rice'],
    },
    'NPK 10-26-26': {
        'temperature': (28, 4),
        'humidity':    (54, 7),
        'moisture':    (44, 6),
        'nitrogen':    (48, 7),
        'phosphorus':  (54, 7),     # Balanced P+K
        'potassium':   (55, 7),
        'soil_pref':   ['Red', 'Loamy', 'Sandy'],
        'crop_pref':   ['Rice', 'Wheat', 'Maize', 'Cotton'],
    },
    'NPK 20-20-20': {
        'temperature': (31, 4),
        'humidity':    (66, 6),
        'moisture':    (54, 6),
        'nitrogen':    (56, 7),     # Balanced all three
        'phosphorus':  (43, 6),
        'potassium':   (40, 6),
        'soil_pref':   ['Loamy', 'Sandy', 'Red'],
        'crop_pref':   ['Sugarcane', 'Cotton', 'Maize', 'Tobacco'],
    },
    'SSP': {
        'temperature': (26, 4),
        'humidity':    (42, 7),
        'moisture':    (32, 6),
        'nitrogen':    (10, 6),
        'phosphorus':  (52, 7),     # Moderate P for oilseeds
        'potassium':   (12, 6),
        'soil_pref':   ['Sandy', 'Red', 'Loamy'],
        'crop_pref':   ['Ground Nuts', 'Oil Seeds', 'Pulses'],
    },
    'Ammonium Sulphate': {
        'temperature': (28, 4),
        'humidity':    (50, 7),
        'moisture':    (40, 6),
        'nitrogen':    (24, 7),
        'phosphorus':  (22, 7),
        'potassium':   (28, 7),
        'soil_pref':   ['Red', 'Sandy', 'Loamy'],
        'crop_pref':   ['Tobacco', 'Millets', 'Maize', 'Wheat'],
    },
}

# ─── Generate data ──────────────────────────────────────────────────────────
rows_per_class = TOTAL_ROWS // len(PROFILES)
extra = TOTAL_ROWS - rows_per_class * len(PROFILES)

all_rows = []

for idx, (fert_name, profile) in enumerate(PROFILES.items()):
    n = rows_per_class + (1 if idx < extra else 0)

    temp = np.random.normal(*profile['temperature'], n).clip(-10, 60).round(1)
    hum  = np.random.normal(*profile['humidity'], n).clip(0, 100).round(0).astype(int)
    mois = np.random.normal(*profile['moisture'], n).clip(0, 100).round(0).astype(int)
    nit  = np.random.normal(*profile['nitrogen'], n).clip(0, 200).round(0).astype(int)
    pho  = np.random.normal(*profile['phosphorus'], n).clip(0, 200).round(0).astype(int)
    pot  = np.random.normal(*profile['potassium'], n).clip(0, 200).round(0).astype(int)

    # Soil: 70% from preferred, 30% random
    soils = []
    for _ in range(n):
        if np.random.random() < 0.7:
            soils.append(np.random.choice(profile['soil_pref']))
        else:
            soils.append(np.random.choice(SOIL_TYPES))

    # Crop: 70% from preferred, 30% random
    crops = []
    for _ in range(n):
        if np.random.random() < 0.7:
            crops.append(np.random.choice(profile['crop_pref']))
        else:
            crops.append(np.random.choice(CROP_TYPES))

    for i in range(n):
        all_rows.append({
            'Temperature': temp[i],
            'Humidity': hum[i],
            'Moisture': mois[i],
            'Nitrogen': nit[i],
            'Phosphorus': pho[i],
            'Potassium': pot[i],
            'Soil_Type': soils[i],
            'Crop_Type': crops[i],
            'Fertilizer': fert_name,
        })

df = pd.DataFrame(all_rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")
print(f"   Fertilizer distribution:")
for fert, count in df['Fertilizer'].value_counts().sort_index().items():
    print(f"     {fert}: {count}")
