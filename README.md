# Personal Color & Style Recommender

Portfolio-grade end-to-end ML project that analyzes a portrait and recommends:
- clothing color palette
- hair color options
- style, jewelry, and makeup notes

## Project Structure

```text
ai-style/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_images/
├── models/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_clustering.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── face_detection.py
│   ├── feature_extraction.py
│   ├── clustering.py
│   ├── recommender.py
│   ├── visualization.py
│   ├── inference.py
│   └── utils.py
├── train.py
├── inference.py
├── requirements.txt
└── .gitignore
```

## ML Pipeline (ASCII)

```text
Portrait Image
     |
     v
Face Detection (MediaPipe -> OpenCV fallback)
     |
     v
Face Crop
     |
     v
Feature Extraction
(LAB + HSV + brightness + contrast + saturation + dominant hair color)
     |
     v
Preprocessing (validate -> impute -> scale)
     |
     v
KMeans Clustering (k=3..6, silhouette-based auto-selection)
     |
     v
Semantic Labeling + Recommendation Engine
     |
     v
CLI / Streamlit Output
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1) Train

Put portrait images into `data/raw/`, then run:

```bash
python train.py
```

Artifacts saved to `models/`:
- `scaler.pkl`
- `cluster_model.pkl`
- `metadata.json`

### 2) CLI Inference

```bash
python inference.py path/to/photo.jpg
```

### 3) Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Methodology

- **Face detection**: MediaPipe with OpenCV Haar fallback
- **Feature engineering**:
  - average skin tone in LAB
  - average skin HSV
  - brightness, contrast, saturation
  - dominant hair HSV using KMeans on top crop region
- **Preprocessing**: feature validation, median imputation, `StandardScaler`
- **Modeling**: KMeans with auto `k` from 3 to 6 via silhouette score
- **Post-processing**: semantic cluster names + recommendation dictionaries + explanation text

## Results

The system outputs:
- predicted style cluster
- explanation string (why this cluster fits)
- clothing palette cards
- hair/jewelry/makeup/style suggestions
- feature visualizations

## Future Improvements

- add eye color estimation with iris segmentation
- improve skin masking (landmark-based)
- train on larger curated dataset for more stable clusters
- support multiple faces and ranking
- add model monitoring and drift checks
