import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="🔍 Deepfake Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .real-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_detection_model():
    """Load the trained model"""
    try:
        model = load_model('models/deepfake_detector.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Make sure you've trained the model first by running: python src/train.py")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL to numpy array
    img = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize to 128x128
    img = cv2.resize(img, (128, 128))
    
    # Normalize to 0-1
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(image, model):
    """Make prediction on image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    return prediction

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Deepfake Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Image Authentication System</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_detection_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This application uses a Convolutional Neural Network (CNN) 
        to detect deepfake or AI-generated images.
        
        **Model Performance:**
        - Test Accuracy: 99.75%
        - Architecture: Custom CNN
        - Parameters: ~8.9M
        """)
        
        st.header("📊 How It Works")
        st.markdown("""
        1. **Upload** an image of a face
        2. **CNN analyzes** facial features
        3. **Model predicts** Real vs Fake
        4. **Results** shown with confidence
        """)
        
        st.header("💡 Tips")
        st.warning("""
        ✅ Use clear, well-lit face images  
        ✅ Frontal faces work best  
        ✅ Avoid heavily filtered images  
        ❌ Group photos may not work well  
        """)
        
        st.header("📈 Statistics")
        st.metric("Model Accuracy", "99.75%")
        st.metric("Total Parameters", "8.9M")
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload an image to analyze",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    # Example images section
    with st.expander("🖼️ Don't have an image? Try our examples"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Upload a real face photo")
        with col2:
            st.info("Or an AI-generated image")
        with col3:
            st.info("And see the results!")
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner('🧠 Analyzing image...'):
                    # Add progress bar for effect
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Make prediction
                    prediction = predict_image(image, model)
                    
                    # Clear progress bar
                    progress_bar.empty()
                
                # Display results
                st.markdown("---")
                
                if prediction > 0.5:
                    # FAKE IMAGE
                    confidence = prediction * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box fake-box">
                        <h1 style="color: #dc3545; margin: 0;">🚨 FAKE IMAGE DETECTED</h1>
                        <h2 style="color: #721c24; margin-top: 1rem;">Confidence: {confidence:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("⚠️ This image appears to be AI-generated or manipulated.")
                    
                    # Additional info
                    with st.expander("📊 Detailed Analysis"):
                        st.write(f"**Raw Prediction Score:** {prediction:.6f}")
                        st.write(f"**Threshold:** 0.5")
                        st.write(f"**Classification:** FAKE (score > 0.5)")
                        
                        # Confidence meter
                        st.progress(float(prediction))
                        
                else:
                    # REAL IMAGE
                    confidence = (1 - prediction) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box real-box">
                        <h1 style="color: #28a745; margin: 0;">✅ REAL IMAGE</h1>
                        <h2 style="color: #155724; margin-top: 1rem;">Confidence: {confidence:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ This image appears to be authentic.")
                    
                    # Additional info
                    with st.expander("📊 Detailed Analysis"):
                        st.write(f"**Raw Prediction Score:** {prediction:.6f}")
                        st.write(f"**Threshold:** 0.5")
                        st.write(f"**Classification:** REAL (score < 0.5)")
                        
                        # Confidence meter
                        st.progress(float(1 - prediction))
    
    else:
        # Instructions when no image uploaded
        st.info("👆 Please upload an image to begin analysis")
        
        # Feature highlights
        st.markdown("---")
        st.subheader("✨ Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Accurate
            99.75% test accuracy on validation data
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ Fast
            Results in seconds using deep learning
            """)
        
        with col3:
            st.markdown("""
            ### 🔒 Secure
            All processing done locally
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>Made with ❤️ using TensorFlow & Streamlit</p>
        <p style="font-size: 0.8rem;">⚠️ This tool is for educational purposes. Always verify important information through multiple sources.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()