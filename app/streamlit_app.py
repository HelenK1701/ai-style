"""
Streamlit UI for Personal Color & Style Recommender.
"""

from __future__ import annotations

import tempfile

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import run_inference
from src.utils import bgr_to_pil
from src.visualization import create_color_swatch, create_feature_radar

st.set_page_config(page_title="Personal Color & Style Recommender", page_icon="🎨", layout="wide")
st.title("🎨 Personal Color & Style Recommender")
st.caption("Upload a portrait and get ML-based color and style recommendations.")

uploaded = st.file_uploader("Upload portrait image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)

with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp:
    image.save(temp.name)
    result = run_inference(temp.name)

left, right = st.columns(2)
with left:
    st.subheader("Detected face crop")
    st.image(bgr_to_pil(result["face_crop_rgb"][:, :, ::-1]), use_column_width=True)
with right:
    st.subheader("Predicted style cluster")
    st.success(result["cluster_label"])
    st.write(result["explanation"])

st.subheader("Extracted features")
feature_df = pd.DataFrame([result["features"]])
st.dataframe(feature_df, use_container_width=True)

st.subheader("Recommended palette")
palette = result["recommendations"].get("clothing_palette", [])
if palette:
    fig = create_color_swatch([str(c) for c in palette])
    st.pyplot(fig)

st.subheader("Recommendations")
rec = result["recommendations"]
st.write("**Hair colors:**", ", ".join(rec.get("hair_colors", [])))
st.write("**Jewelry:**", ", ".join(rec.get("jewelry", [])))
st.write("**Makeup tones:**", ", ".join(rec.get("makeup_tones", [])))
st.write("**Style notes:**", rec.get("style_notes", ""))

st.subheader("Feature radar")
st.pyplot(create_feature_radar(result["features"]))
