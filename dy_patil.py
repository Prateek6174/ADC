
# ---------------------------
# Streamlit SSD300 Object Detection App
# Downloads checkpoint from Google Drive using gdown (first run only)
# Robustly loads either checkpoint['model'] OR a raw state_dict
# Works with PyTorch >= 2.6 by explicitly setting weights_only=False
# ---------------------------

import os
import io
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import gdown

# --- Import your local modules ---
# These files must be present in the same repo/folder.
from model import SSD300
from utils import label_map, rev_label_map, label_color_map

# ---------------------------
# Page & layout settings
# ---------------------------
st.set_page_config(
    page_title="SSD300 Object Detection",
    page_icon="🟦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Auto Defect Classifier - Beta Version")
st.caption(
    "Upload an image, run detection, and download the annotated result. "
    "Model checkpoint is fetched from Google Drive automatically."
)

# ---------------------------
# Device configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Running on **{device.type.upper()}**")

# ---------------------------
# Image transforms (match training/pretrained VGG norms)
# ---------------------------
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# ---------------------------
# Google Drive checkpoint source (YOUR FILE)
# ---------------------------
# Your share link: https://drive.google.com/file/d/1CN-zQooT2NE-BcbR2vGNFb19NEBPvAXz/view?usp=drive_link
GDRIVE_FILE_ID = "1CN-zQooT2NE-BcbR2vGNFb19NEBPvAXz"  # <-- YOUR actual file ID
CHECKPOINT_LOCAL = "checkpoint_ssd300.pth"            # local filename

# ---------------------------
# Helper: build model safely from state_dict (handles 'module.' prefixes)
# ---------------------------
def build_model_from_state_dict(state_dict: dict) -> torch.nn.Module:
    model = SSD300(n_classes=len(label_map)).to(device)

    # Some checkpoints saved via DataParallel have 'module.' prefix in keys
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Try strict load first, fall back to non-strict with a warning
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        st.warning("Strict load failed—trying non‑strict load. "
                   "This usually means the checkpoint keys differ slightly.")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

# ---------------------------
# Model loader (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Downloads the checkpoint from Google Drive (if not present),
    loads the SSD300 model from the checkpoint, and caches the result.
    Supports:
      - checkpoint['model'] saved as a whole model object
      - raw state_dict (optionally under checkpoint['state_dict'])
    NOTE: PyTorch 2.6 default weights_only=True breaks unpickling model objects.
          We set weights_only=False explicitly since you trust your own checkpoint.
    """
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    # Download once if file not found (or incomplete)
    if not os.path.exists(CHECKPOINT_LOCAL) or os.path.getsize(CHECKPOINT_LOCAL) < 10_000:
        with st.spinner("Downloading model checkpoint from Google Drive..."):
            gdown.download(url, CHECKPOINT_LOCAL, quiet=False)

    # Load checkpoint from disk
    try:
        # Critical fix: allow full unpickling of your saved model object
        ckpt = torch.load(CHECKPOINT_LOCAL, map_location=device, weights_only=False)
    except Exception as e:
        st.error("❌ Failed to read the checkpoint file. "
                 "Please re-check the Google Drive link sharing (Anyone with the link).")
        st.exception(e)
        raise

    # Case 1: full checkpoint dict with a 'model' entry (your case)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"].to(device)
        model.eval()
        return model

    # Case 2: checkpoint with 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return build_model_from_state_dict(ckpt["state_dict"])

    # Case 3: raw state_dict directly
    if isinstance(ckpt, dict):
        return build_model_from_state_dict(ckpt)

    # If it's not a dict (rare), assume torch-scripted whole model
    if hasattr(ckpt, "eval"):
        model = ckpt.to(device)
        model.eval()
        return model

    # Fallback error
    raise RuntimeError("Unsupported checkpoint format. Expected a dict with 'model' or 'state_dict', "
                       "or a raw state_dict.")

# Try loading the model (and download if needed)
try:
    model = load_model()
    st.success("Model loaded ✅")
except Exception:
    st.stop()

# ---------------------------
# Detection utilities
# ---------------------------
def detect_objects(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200):
    """
    Run SSD300 detection on a single PIL image.
    Returns scaled boxes (to original size), string labels, and scores.
    """
    # Preprocess
    image = normalize(to_tensor(resize(original_image))).to(device)

    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))
        det_boxes, det_labels, det_scores = model.detect_objects(
            predicted_locs,
            predicted_scores,
            min_score=min_score,
            max_overlap=max_overlap,
            top_k=top_k,
        )

    # Convert fractional box coords to original pixel coords
    det_boxes = det_boxes[0].to("cpu")
    original_dims = torch.FloatTensor([
        original_image.width, original_image.height,
        original_image.width, original_image.height
    ]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Labels & scores
    det_labels_tensor = det_labels[0].to("cpu")
    det_scores = det_scores[0].to("cpu")
    det_label_names = [rev_label_map[l] for l in det_labels_tensor.tolist()]

    return det_boxes, det_label_names, det_scores


def draw_detections(original_image, det_boxes, det_labels, det_scores):
    """
    Draw bounding boxes with class name and confidence on the image.
    """
    annotated = original_image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try a common font; fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i in range(det_boxes.size(0)):
        cls_name = det_labels[i]
        if cls_name == "background":
            continue

        box = det_boxes[i].tolist()
        # Draw a slightly thicker border by repeated rectangles
        for offset in range(3):
            draw.rectangle(
                [x + offset for x in box],
                outline=label_color_map.get(cls_name, "#0071e3"),
                width=1
            )

        score = det_scores[i].item()
        text = f"{cls_name.upper()} {score:.0%}"

        # Compute text box size & positions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_xy = [box[0] + 4, box[1] - text_h - 6]
        textbox = [box[0], box[1] - text_h - 6, box[0] + text_w + 8, box[1]]

        draw.rectangle(textbox, fill=label_color_map.get(cls_name, "#0071e3"))
        draw.text(text_xy, text, fill="white", font=font)

    del draw
    return annotated


def to_png_bytes(pil_image):
    """
    Convert a PIL image to PNG bytes for download.
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Sidebar (optional advanced settings)
# ---------------------------
with st.sidebar:
    st.subheader("Advanced Settings")
    min_score = st.slider("Minimum score (keep boxes ≥)", 0.0, 1.0, 0.2, 0.01)
    max_overlap = st.slider("Max overlap (NMS IoU)", 0.0, 1.0, 0.5, 0.01)
    top_k = st.number_input("Top‑K keep (across classes)", min_value=1, max_value=1000, value=200, step=10)

# ---------------------------
# Main UI
# ---------------------------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

# Load and show the original image
image = Image.open(uploaded).convert("RGB")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(image, use_column_width=True)

# Detect button
run = st.button("🔎 Detect Objects")

if run:
    with st.spinner("Running detection..."):
        det_boxes, det_labels, det_scores = detect_objects(
            model, image, min_score=min_score, max_overlap=max_overlap, top_k=top_k
        )

    # If only background or no detections, show original
    if det_labels == ["background"] or len(det_labels) == 0:
        annotated = image
        stats = None
        st.warning("No objects detected (background).")
    else:
        annotated = draw_detections(image, det_boxes, det_labels, det_scores)
        unique = sorted(set([l for l in det_labels if l != "background"]))
        n_objects = len([l for l in det_labels if l != "background"])
        stats = {
            "n_objects": n_objects,
            "unique_classes": len(unique),
            "detected": unique
        }

    # Show annotated result
    with col2:
        st.subheader("Detection Result")
        st.image(annotated, use_column_width=True)

    # Show quick stats
    if stats:
        st.success(
            f"**Objects:** {stats['n_objects']}  |  "
            f"**Classes:** {stats['unique_classes']}  |  "
            f"**Detected:** {', '.join(stats['detected'])}"
        )

    # Download button
    st.download_button(
        label="⬇️ Download Annotated Image (PNG)",
        data=to_png_bytes(annotated),
        file_name="result.png",
        mime="image/png",
    )
else:
    with col2:
        st.subheader("Detection Result")
        st.info("Click **Detect Objects** to analyze the image.")
