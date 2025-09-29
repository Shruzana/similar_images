import streamlit as st
import numpy as np
import os
import zipfile
import requests
from io import BytesIO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from PIL import Image

# -----------------------------
# Download and extract dataset
# -----------------------------
ZIP_URL = "https://github.com/Shruzana/similar_images/raw/main/similar_images.zip"
IMAGE_DIR = "images"
FEAT_CACHE = "features.npy"
NAME_CACHE = "filenames.npy"
TOP_N = 5

if not os.path.exists(IMAGE_DIR):
    st.write("📥 Downloading and extracting image dataset...")
    response = requests.get(ZIP_URL)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(IMAGE_DIR)

# -----------------------------
# Feature extractor
# -----------------------------
@st.cache_resource
def get_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

def extract_features(file_path, model):
    img = load_img(file_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feature = model.predict(arr, verbose=0)
    return feature.flatten()

# -----------------------------
# Recursively collect all image files
# -----------------------------
def get_all_image_files(root_dir):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = []
    for subdir, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(valid_ext):
                files.append(os.path.join(subdir, f))
    return sorted(files)

# -----------------------------
# Compute or load features
# -----------------------------
def compute_and_cache_features(image_dir, model):
    filenames = get_all_image_files(image_dir)
    if len(filenames) == 0:
        st.error(f"No image files found in {image_dir} or its subfolders.")
        return [], []
    features = []
    for fn in stqdm(filenames, "Extracting folder features"):
        features.append(extract_features(fn, model))
    features = normalize(np.array(features), axis=1)
    np.save(FEAT_CACHE, features)
    np.save(NAME_CACHE, filenames)
    return filenames, features

@st.cache_data
def load_image_features(image_dir, model):
    if os.path.exists(FEAT_CACHE) and os.path.exists(NAME_CACHE):
        filenames = np.load(NAME_CACHE, allow_pickle=True)
        features = np.load(FEAT_CACHE)
        return filenames, features
    else:
        return compute_and_cache_features(image_dir, model)

# -----------------------------
# Similar image search
# -----------------------------
def find_similar_images(query_img_path, features_db, filenames_db, model, top_n=TOP_N):
    qf = extract_features(query_img_path, model)
    qf = normalize([qf])[0]
    sims = cosine_similarity([qf], features_db)[0]
    top_idx = np.argsort(-sims)[:top_n]
    return [(filenames_db[i], sims[i]) for i in top_idx]

# -----------------------------
# tqdm for progress bar
# -----------------------------
def stqdm(iterable, desc):
    progress = st.progress(0)
    items = list(iterable)
    n = len(items)
    for i, item in enumerate(items):
        progress.progress((i + 1) / n, text=f"{desc}: {i + 1}/{n}")
        yield item
    progress.empty()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Find Similar Images")
st.markdown("Upload or select an image. The app will show the most visually similar images from the database.")

model = get_feature_extractor()
image_files, image_feats = load_image_features(IMAGE_DIR, model)

if len(image_files) == 0:
    st.stop()

img_display_names = [os.path.basename(fp) for fp in image_files]
option = st.selectbox("Select a query image from the database:", img_display_names)
selected_path = image_files[list(img_display_names).index(option)]

uploaded = st.file_uploader("Or upload a new image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
if uploaded:
    query_img = Image.open(uploaded).convert('RGB')
    os.makedirs('temp', exist_ok=True)
    query_path = os.path.join('temp', uploaded.name)
    query_img.save(query_path)
    query_to_use = query_path
else:
    query_img = Image.open(selected_path)
    query_to_use = selected_path

st.subheader("Query Image")
st.image(query_img, width=224)

results = find_similar_images(query_to_use, image_feats, image_files, model, top_n=TOP_N)

st.subheader("Top Similar Images")
cols = st.columns(TOP_N)
for idx, (img_path, score) in enumerate(results):
    with cols[idx]:
        st.image(Image.open(img_path), caption=f"Score: {score * 100:.1f}%", use_container_width=True)

if uploaded:
    os.remove(query_path)
