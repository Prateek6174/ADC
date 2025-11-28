import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import time

# Import from the repository files
from model import SSD300
from utils import label_map, rev_label_map, label_color_map

# Page configuration
st.set_page_config(
    page_title="Object Detection",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apple-authentic CSS with single page layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, .stApp, .main, .block-container {
        background: #000000 !important;
        height: 100vh !important;
        overflow: hidden !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .main {
        padding: 0 !important;
    }
    
    .block-container {
        padding: 44px 0 0 0 !important;
        max-width: 100% !important;
        height: calc(100vh - 44px) !important;
    }
    
    /* Apple navigation */
    .apple-nav {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        height: 44px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .nav-content {
        max-width: 1400px;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 40px;
    }
    
    .nav-logo {
        color: #f5f5f7;
        font-size: 14px;
        font-weight: 600;
    }
    
    .nav-item {
        color: #f5f5f7;
        font-size: 12px;
        font-weight: 400;
        opacity: 0.8;
        margin-left: 32px;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 80px 40px 60px 40px;
    }
    
    .hero-title {
        font-size: 72px;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.015em;
        background: linear-gradient(135deg, #ffffff 0%, #0071e3 50%, #00c7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
    }
    
    .hero-subtitle {
        font-size: 21px;
        line-height: 1.4;
        font-weight: 400;
        color: rgba(245, 245, 247, 0.8);
        margin-bottom: 40px;
    }
    
    /* Upload container */
    .upload-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* File uploader */
    .stFileUploader {
        width: 100%;
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(0, 113, 227, 0.5) !important;
        border-radius: 20px !important;
        padding: 60px 40px !important;
        transition: all 0.3s !important;
    }
    
    .stFileUploader > div:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(0, 113, 227, 0.8) !important;
    }
    
    .stFileUploader section[data-testid="stFileUploaderDropzone"] + div {
        display: none !important;
    }
    
    .stFileUploader label {
        color: rgba(245, 245, 247, 0.6) !important;
        font-size: 17px !important;
        font-weight: 400 !important;
    }
    
    /* Info box next to home button */
    .info-box {
        position: fixed;
        top: 54px;
        right: 120px;
        z-index: 9998;
        background: rgba(0, 113, 227, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 113, 227, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        color: #f5f5f7;
        font-size: 13px;
        display: flex;
        gap: 20px;
        align-items: center;
    }
    
    .info-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .info-value {
        font-size: 20px;
        font-weight: 700;
        color: #0071e3;
    }
    
    .info-label {
        font-size: 11px;
        color: rgba(245, 245, 247, 0.6);
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    .info-classes {
        font-size: 12px;
        color: rgba(245, 245, 247, 0.8);
        padding-left: 20px;
        border-left: 1px solid rgba(255, 255, 255, 0.2);
        max-width: 300px;
    }
    
    /* Home button */
    .home-btn {
        position: fixed;
        top: 54px;
        right: 40px;
        z-index: 9998;
    }
    
    .home-btn button {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        color: #f5f5f7 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 10px 20px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        width: auto !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    /* Columns layout */
    [data-testid="column"] {
        padding: 0 10px !important;
        height: calc(100vh - 84px) !important;
        overflow: hidden !important;
    }
    
    /* Card container */
    .card-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(40px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 20px;
        text-align: center;
        flex-shrink: 0;
    }
    
    .card-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        min-height: 0;
    }
    
    .image-wrapper {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        min-height: 0;
        margin-bottom: 20px;
    }
    
    /* Images */
    .stImage {
        max-height: 100% !important;
        max-width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stImage img {
        max-height: 100% !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 113, 227, 0.3) !important;
    }
    
    /* Button */
    .stButton {
        flex-shrink: 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0071e3 0%, #00a8ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
        width: 100% !important;
        margin: 0 !important;
        box-shadow: 0 8px 24px rgba(0, 113, 227, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(0, 113, 227, 0.6) !important;
    }
    
    /* Stats */
    .stats-box {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        display: flex;
        justify-content: space-around;
        gap: 20px;
        flex-shrink: 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        color: #0071e3;
        margin-bottom: 4px;
    }
    
    .stat-label {
        font-size: 12px;
        color: rgba(245, 245, 247, 0.6);
        text-transform: uppercase;
    }
    
    /* Empty state */
    .empty-state {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: rgba(245, 245, 247, 0.4);
        font-size: 17px;
        min-height: 0;
    }
    
    /* Loading animation */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(20px);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .scanner {
        width: 300px;
        height: 300px;
        position: relative;
        margin-bottom: 40px;
    }
    
    .scan-line {
        position: absolute;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #0071e3, #00c7ff, #0071e3, transparent);
        box-shadow: 0 0 20px #0071e3;
        animation: scan 2s ease-in-out infinite;
    }
    
    @keyframes scan {
        0%, 100% { top: 0; opacity: 0; }
        10%, 90% { opacity: 1; }
        100% { top: 100%; }
    }
    
    .corner {
        position: absolute;
        width: 40px;
        height: 40px;
        border: 3px solid #0071e3;
    }
    
    .corner.tl { top: 0; left: 0; border-right: none; border-bottom: none; }
    .corner.tr { top: 0; right: 0; border-left: none; border-bottom: none; }
    .corner.bl { bottom: 0; left: 0; border-right: none; border-top: none; }
    .corner.br { bottom: 0; right: 0; border-left: none; border-top: none; }
    
    .loading-text {
        font-size: 24px;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 10px;
    }
    
    .loading-subtext {
        font-size: 15px;
        color: rgba(245, 245, 247, 0.6);
    }
    
    /* Alert */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f5f5f7 !important;
    }
    
    /* Hide scrollbar and streamlit elements */
    ::-webkit-scrollbar { display: none; }
    #MainMenu, footer, .stDeployButton, header { display: none !important; }
    .stSpinner { display: none !important; }
    
    /* Remove any extra padding/margin */
    .element-container {
        margin: 0 !important;
    }
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def load_model():
    """Load the SSD300 model from checkpoint"""
    try:
        checkpoint_path = "checkpoint_ssd300.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = checkpoint['model']
        model = model.to(device)
        model.eval()
        return model, True
    except Exception as e:
        return None, False

def find_intersection_boxes(box1, box2):
    """Calculate intersection area between two boxes"""
    lower_bounds = torch.max(box1[:, :2], box2[:, :2])
    upper_bounds = torch.min(box1[:, 2:], box2[:, 2:])
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, 0] * intersection_dims[:, 1]

def detect_objects(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200):
    """Detect objects in an image"""
    image = normalize(to_tensor(resize(original_image))).to(device)
    
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))
    
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=min_score, max_overlap=max_overlap, top_k=top_k
    )
    
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor([
        original_image.width, original_image.height,
        original_image.width, original_image.height
    ]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    
    det_labels_tensor = det_labels[0].to('cpu')
    det_scores = det_scores[0].to('cpu')
    
    if len(det_boxes) > 1:
        keep_indices = []
        sorted_indices = torch.argsort(det_scores, descending=True)
        
        for i in sorted_indices:
            if det_labels_tensor[i] == 0:
                continue
                
            should_keep = True
            box_i = det_boxes[i].unsqueeze(0)
            
            for kept_idx in keep_indices:
                box_kept = det_boxes[kept_idx].unsqueeze(0)
                intersection = find_intersection_boxes(box_i, box_kept)
                area_i = (box_i[0, 2] - box_i[0, 0]) * (box_i[0, 3] - box_i[0, 1])
                area_kept = (box_kept[0, 2] - box_kept[0, 0]) * (box_kept[0, 3] - box_kept[0, 1])
                union = area_i + area_kept - intersection
                iou = intersection / union
                
                if iou > max_overlap:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i.item())
        
        if len(keep_indices) > 0:
            keep_indices = torch.tensor(keep_indices)
            det_boxes = det_boxes[keep_indices]
            det_labels_tensor = det_labels_tensor[keep_indices]
            det_scores = det_scores[keep_indices]
    
    det_labels_names = [rev_label_map[l] for l in det_labels_tensor.tolist()]
    return det_boxes, det_labels_names, det_scores

def draw_detections(original_image, det_boxes, det_labels, det_scores):
    """Draw bounding boxes and labels on image"""
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    for i in range(det_boxes.size(0)):
        cls_name = det_labels[i]
        
        if cls_name == 'background':
            continue
        
        box_location = det_boxes[i].tolist()
        
        for offset in range(3):
            draw.rectangle(
                xy=[l + offset for l in box_location],
                outline=label_color_map.get(cls_name, '#0071e3'),
                width=1
            )
        
        score = det_scores[i].item()
        text = f'{cls_name.upper()} {score:.0%}'
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_location = [box_location[0] + 4, box_location[1] - text_h - 6]
        textbox_location = [
            box_location[0], box_location[1] - text_h - 6,
            box_location[0] + text_w + 8, box_location[1]
        ]
        
        draw.rectangle(xy=textbox_location, fill=label_color_map.get(cls_name, '#0071e3'))
        draw.text(xy=text_location, text=text, fill='white', font=font)
    
    del draw
    return annotated_image

def reset_app():
    """Reset the application"""
    st.session_state.uploaded_image = None
    st.session_state.detection_done = False
    st.session_state.annotated_image = None
    st.session_state.stats = None
    st.session_state.processing = False

def main():
    # Navigation
    st.markdown("""
        <div class='apple-nav'>
            <div class='nav-content'>
                <div class='nav-logo'>AI Vision</div>
                <div>
                    <span class='nav-item'>Technology</span>
                    <span class='nav-item'>Neural Networks</span>
                    <span class='nav-item'>Documentation</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, success = load_model()
    
    if not success:
        st.error("Unable to load model checkpoint.")
        return
    
    # Show loading animation
    if st.session_state.processing:
        st.markdown("""
            <div class='loading-overlay'>
                <div class='scanner'>
                    <div class='corner tl'></div>
                    <div class='corner tr'></div>
                    <div class='corner bl'></div>
                    <div class='corner br'></div>
                    <div class='scan-line'></div>
                </div>
                <div class='loading-text'>ANALYZING IMAGE</div>
                <div class='loading-subtext'>NEURAL NETWORK PROCESSING</div>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(2)
        
        try:
            det_boxes, det_labels, det_scores = detect_objects(model, st.session_state.uploaded_image)
            
            if det_labels == ['background'] or len(det_labels) == 0:
                st.session_state.annotated_image = st.session_state.uploaded_image
                st.session_state.stats = None
            else:
                annotated_image = draw_detections(st.session_state.uploaded_image, det_boxes, det_labels, det_scores)
                st.session_state.annotated_image = annotated_image
                
                unique_objects = sorted(set([l for l in det_labels if l != 'background']))
                n_objects = len([l for l in det_labels if l != 'background'])
                
                st.session_state.stats = {
                    'n_objects': n_objects,
                    'unique_classes': len(unique_objects),
                    'detected': unique_objects
                }
            
            st.session_state.detection_done = True
            st.session_state.processing = False
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.processing = False
    
    # Main content
    if st.session_state.uploaded_image is None:
        # Landing page with upload
        st.markdown("""
            <div class='hero-section'>
                <h1 class='hero-title'>Object Detection</h1>
                <p class='hero-subtitle'></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert('RGB')
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Show info box next to home button after detection
        if st.session_state.detection_done and st.session_state.stats:
            classes_str = ", ".join(st.session_state.stats['detected'][:3])
            if len(st.session_state.stats['detected']) > 3:
                classes_str += f" +{len(st.session_state.stats['detected']) - 3} more"
            
            st.markdown(f"""
                <div class='info-box'>
                    <div class='info-item'>
                        <div class='info-value'>{st.session_state.stats['n_objects']}</div>
                        <div class='info-label'>Objects</div>
                    </div>
                    <div class='info-item'>
                        <div class='info-value'>{st.session_state.stats['unique_classes']}</div>
                        <div class='info-label'>Classes</div>
                    </div>
                    <div class='info-classes'>{classes_str}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Home button
        st.markdown("<div class='home-btn'>", unsafe_allow_html=True)
        if st.button("Home", key="home_btn"):
            reset_app()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Split screen view with cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card-container'><div class='card-title'>Original Image</div><div class='card-content'>", unsafe_allow_html=True)
            
            st.markdown("<div class='image-wrapper'>", unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if not st.session_state.detection_done:
                st.button("Detect Objects", key="detect")
                if st.session_state.get('detect'):
                    st.session_state.processing = True
                    st.rerun()
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card-container'><div class='card-title'>Detection Result</div><div class='card-content'>", unsafe_allow_html=True)
            
            if st.session_state.detection_done and st.session_state.annotated_image is not None:
                st.markdown("<div class='image-wrapper'>", unsafe_allow_html=True)
                st.image(st.session_state.annotated_image, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.session_state.stats:
                    st.markdown(f"""
                        <div class='stats-box'>
                            <div class='stat-item'>
                                <div class='stat-value'>{st.session_state.stats['n_objects']}</div>
                                <div class='stat-label'>Objects</div>
                            </div>
                            <div class='stat-item'>
                                <div class='stat-value'>{st.session_state.stats['unique_classes']}</div>
                                <div class='stat-label'>Classes</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='empty-state'>Click 'Detect Objects' to analyze the image</div>", unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()