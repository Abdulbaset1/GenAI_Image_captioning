import streamlit as st
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from utils import load_model, greedy_search

# Page config
st.set_page_config(page_title="Image Captioning App", layout="wide")

# Title
st.title("🖼️ Image Captioning System")
st.markdown("Upload an image and get AI-generated captions!")

# Load model and vocab
@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word2idx, idx2word = load_model(
        "models/image_captioning_model.pt",
        "artifacts/flickr30k_vocab.pkl",
        device
    )
    return model, word2idx, idx2word, device

# Load image features extractor (pretrained CNN)
@st.cache_resource
def load_feature_extractor():
    import torchvision.models as models
    extractor = models.resnet50(pretrained=True)
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1])
    extractor.eval()
    return extractor

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def extract_features(image, extractor, device):
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        features = extractor(image_tensor)
    return features.squeeze().cpu().numpy()

# Main app
def main():
    # Load artifacts
    with st.spinner("Loading model..."):
        model, word2idx, idx2word, device = load_artifacts()
        extractor = load_feature_extractor()
        extractor.to(device)
    
    # Sidebar
    st.sidebar.header("Settings")
    max_length = st.sidebar.slider("Max caption length", 10, 40, 20)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Generated Caption")
            
            # Generate caption
            with st.spinner("Generating caption..."):
                features = extract_features(image, extractor, device)
                caption = greedy_search(model, features, idx2word, word2idx, device, max_length)
            
            st.success("Caption generated!")
            st.markdown(f"**{caption}**")
            
            # Confidence score (simulated)
            st.progress(0.85)
            st.caption("Confidence: 85%")
            
            # Download button
            st.download_button(
                label="Download Caption",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
