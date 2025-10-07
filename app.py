import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# ----------------------
# Config
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
IMAGE_FOLDER = "images"

st.set_page_config(page_title="Image → Image Search", page_icon="🔍", layout="wide")

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor

model, processor = load_model()

# ----------------------
# Candidate labels (better prompts)
# ----------------------
LABELS = [
    "a domestic cat", "a house cat", "a pet cat",
    "a dog", "a pet dog",
    "a lion", "a wild lion",
    "a tiger", "a wild tiger",
    "a giraffe", "a zebra", "an elephant", "a monkey"
]

@st.cache_resource
def compute_label_embeddings():
    inputs = processor(text=LABELS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    return text_emb.cpu().numpy()

label_embeddings = compute_label_embeddings()

# ----------------------
# Classify image → best label
# ----------------------
def classify_image(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    emb = emb.cpu().numpy()

    sims = label_embeddings @ emb.T
    best_idx = int(np.argmax(sims))
    return LABELS[best_idx]

# ----------------------
# Precompute dataset embeddings + labels
# ----------------------
@st.cache_resource
def compute_dataset_embeddings_and_labels():
    embeddings, filepaths, labels = [], [], []
    for fname in os.listdir(IMAGE_FOLDER):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(IMAGE_FOLDER, fname)
            img = Image.open(path).convert("RGB")

            # compute embedding
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy())
            filepaths.append(path)

            # precompute label
            labels.append(classify_image(img))

    return np.vstack(embeddings), filepaths, labels

dataset_embeddings, dataset_files, dataset_labels = compute_dataset_embeddings_and_labels()

# ----------------------
# Similarity function (with hard filtering)
# ----------------------
def search_similar_images(query_img, top_k=5):
    # classify query image
    query_label = classify_image(query_img)

    # find dataset indices with same class
    valid_idxs = [i for i, lbl in enumerate(dataset_labels) if lbl == query_label]

    if not valid_idxs:  # fallback: search all
        valid_idxs = list(range(len(dataset_files)))

    # embed query
    inputs = processor(images=query_img, return_tensors="pt").to(device)
    with torch.no_grad():
        q_emb = model.get_image_features(**inputs)
    q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
    q_emb = q_emb.cpu().numpy()

    # similarity only within filtered class
    sims = (dataset_embeddings[valid_idxs] @ q_emb.T).reshape(-1)  # <-- fix here

    top_idx = np.argsort(-sims)[:top_k]
    return [(dataset_files[valid_idxs[i]], float(sims[i]), query_label) for i in top_idx]

# ----------------------
# UI
# ----------------------
st.title("🔍 Search Images with an Image (Hard-Filtered)")

uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_container_width=True)

    if st.button("Find Similar Images", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            results = search_similar_images(query_image, top_k=5)

        best_label = results[0][2] if results else "unknown"
        st.success(f"Detected category: **{best_label}**")

        cols = st.columns(len(results))
        for col, (path, score, _) in zip(cols, results):
            with col:
                st.image(path, caption=f"Score: {score:.4f}", use_container_width=True)
