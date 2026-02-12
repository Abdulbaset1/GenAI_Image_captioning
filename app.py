import streamlit as st
import torch
import pickle
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms
from utils import load_model, greedy_search
import requests
from pathlib import Path

# Page config
st.set_page_config(page_title="Image Captioning App", layout="wide")

# Title
st.title("🖼️ Image Captioning System")
st.markdown("Upload an image and get AI-generated captions!")

# GitHub release URLs
MODEL_URL = "https://github.com/Abdulbaset1/GenAI_Image_captioning/releases/download/v1/image_captioning_model.pt"
VOCAB_URL = "https://github.com/Abdulbaset1/GenAI_Image_captioning/releases/download/v1/flickr30k_vocab.pkl"

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

@st.cache_resource
def download_artifacts():
    """Download model and vocab from GitHub releases"""
    
    model_path = "models/image_captioning_model.pt"
    vocab_path = "artifacts/flickr30k_vocab.pkl"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... This may take a few minutes..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    # Download vocab if not exists
    if not os.path.exists(vocab_path):
        with st.spinner("Downloading vocabulary..."):
            response = requests.get(VOCAB_URL, stream=True)
            response.raise_for_status()
            
            with open(vocab_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    return model_path, vocab_path

@st.cache_resource
def load_artifacts():
    """Load model and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download files
    model_path, vocab_path = download_artifacts()
    
    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    
    word2idx = vocab_data["word2idx"]
    idx2word = vocab_data["idx2word"]
    
    # Load model
    from model import ImageCaptioningModel
    model = ImageCaptioningModel(vocab_data["vocab_size"])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, word2idx, idx2word, device

@st.cache_resource
def load_feature_extractor():
    """Load pretrained ResNet for feature extraction"""
    import torchvision.models as models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = models.resnet50(pretrained=True)
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1])
    extractor.to(device)
    extractor.eval()
    
    return extractor, device

def preprocess_image(image):
    """Preprocess image for ResNet"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def extract_features(image, extractor, device):
    """Extract features from image using ResNet"""
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        features = extractor(image_tensor)
    return features.squeeze().cpu().numpy()

# Main app
def main():
    st.sidebar.header("About")
    st.sidebar.info(
        "This app generates captions for images using a deep learning model "
        "trained on the Flickr30k dataset."
    )
    
    # Load artifacts
    try:
        with st.spinner("Loading model and vocabulary..."):
            model, word2idx, idx2word, device = load_artifacts()
            extractor, feat_device = load_feature_extractor()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check your internet connection and try again.")
        return
    
    # Sidebar settings
    st.sidebar.header("Settings")
    max_length = st.sidebar.slider("Max caption length", 10, 40, 20)
    
    # Main content
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("📝 Generated Caption")
            
            # Generate caption
            with st.spinner("🎨 Analyzing image and generating caption..."):
                features = extract_features(image, extractor, feat_device)
                caption = greedy_search(model, features, idx2word, word2idx, device, max_length)
            
            # Display caption in nice box
            st.success("✅ Caption generated successfully!")
            
            # Styled caption box
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #ff4b4b;
                    margin: 10px 0;
                ">
                    <h3 style="color: #333; margin: 0;">{caption}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Download button
            st.download_button(
                label="📥 Download Caption",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )
            
            # Try another image button
            if st.button("🔄 Generate Another"):
                st.experimental_rerun()
    
    else:
        # Show sample instructions
        st.info("👆 Please upload an image to generate a caption!")
        
        # Display sample images
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Click on **Browse files** to upload an image
        2. Wait for the model to analyze the image
        3. View the generated caption
        4. Download the caption if needed
        
        **Supported formats:** JPG, JPEG, PNG
        """)

if __name__ == "__main__":
    main()
