import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from body_classifier import BodyTypeClassifier
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Body Type Analyzer",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .body-type-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .measurement-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load and cache the body type classifier"""
    return BodyTypeClassifier()

def display_body_type_info():
    """Display information about body types"""
    st.sidebar.markdown("## 📖 Body Types Guide")
    
    body_types = {
        "Rectangle": "Straight up and down, shoulders ≈ waist ≈ hips",
        "Inverted Triangle": "Broad shoulders, narrow hips",
        "Pear": "Hips wider than shoulders",
        "Hourglass": "Balanced bust and hips, narrow waist",
        "Apple": "Fuller midsection, less defined waist"
    }
    
    for body_type, description in body_types.items():
        st.sidebar.markdown(f"**{body_type}**: {description}")

def create_confidence_bar(confidence):
    """Create a visual confidence bar"""
    confidence_percent = int(confidence * 100)
    color = "#4CAF50" if confidence >= 0.8 else "#FF9800" if confidence >= 0.6 else "#f44336"
    
    return f"""<div style="background: #e0e0e0; border-radius: 10px; height: 25px; margin: 10px 0;">
        <div style="background: {color}; width: {confidence_percent}%; height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {confidence_percent}%
        </div>
    </div>"""

def display_measurements(measurements):
    """Display body measurements in a nice format"""
    st.markdown("### 📏 Body Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="measurement-item">
            <strong>Shoulder-Hip Ratio</strong><br>
            <span style="font-size: 1.5em; color: #667eea;">{measurements['shoulder_hip_ratio']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="measurement-item">
            <strong>Waist-Hip Ratio</strong><br>
            <span style="font-size: 1.5em; color: #667eea;">{measurements['waist_hip_ratio']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="measurement-item">
            <strong>Waist Definition</strong><br>
            <span style="font-size: 1.5em; color: #667eea;">{measurements['waist_definition']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="measurement-item">
            <strong>Torso Length</strong><br>
            <span style="font-size: 1.5em; color: #667eea;">{measurements['torso_length']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

def display_recommendations(recommendations):
    """Display clothing recommendations in an organized way"""
    st.markdown("### 👗 Personalized Style Recommendations")
    
    # Description
    if 'description' in recommendations:
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>Body Type Description</h4>
            <p>{recommendations['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different recommendation categories
    tab1, tab2, tab3, tab4 = st.tabs(["👕 Tops", "👖 Bottoms", "👗 Dresses", "❌ Avoid"])
    
    with tab1:
        if 'tops' in recommendations:
            for item in recommendations['tops']:
                st.markdown(f"✅ {item}")
    
    with tab2:
        if 'bottoms' in recommendations:
            for item in recommendations['bottoms']:
                st.markdown(f"✅ {item}")
    
    with tab3:
        if 'dresses' in recommendations:
            for item in recommendations['dresses']:
                st.markdown(f"✅ {item}")
    
    with tab4:
        if 'avoid' in recommendations:
            for item in recommendations['avoid']:
                st.markdown(f"❌ {item}")
    
    # Styling tips
    if 'styling_tips' in recommendations:
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>💡 Pro Styling Tip</h4>
            <p><em>{recommendations['styling_tips']}</em></p>
        </div>
        """, unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">AI Body Type Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your photo and get personalized fashion recommendations powered by AI</p>', unsafe_allow_html=True)
    
    # Load classifier
    try:
        classifier = load_classifier()
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return
    
    # Sidebar info
    display_body_type_info()
    
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a full-body photo
    2. Ensure you're standing straight
    3. Make sure your entire body is visible
    4. Good lighting helps accuracy
    5. Click 'Analyze Body Type'
    """)
    
    # File upload
    st.markdown("## 📸 Upload Your Photo")
    uploaded_file = st.file_uploader(
        "Choose a full-body image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear, full-body photo where you're standing straight"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Your Photo")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"**Format:** {image.format}")
        
        with col2:
            st.markdown("### Analysis Results")
            
            if st.button("🔍 Analyze Body Type", type="primary", use_container_width=True):
                # Save uploaded file
                temp_path = save_uploaded_file(uploaded_file)
                
                if temp_path:
                    try:
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔄 Processing image...")
                        progress_bar.progress(25)
                        
                        # Analyze body type
                        result = classifier.analyze_body_type(temp_path)
                        progress_bar.progress(75)
                        
                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)
                        
                        # Clean up
                        os.unlink(temp_path)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        if result['success']:
                            # Body type result card
                            st.markdown(f"""
                            <div class="body-type-card">
                                <h2>Your Body Type: {result['body_type']}</h2>
                                <p>Confidence Level</p>
                                {create_confidence_bar(result['confidence'])}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Measurements
                            if result['measurements']:
                                display_measurements(result['measurements'])
                            
                            # Recommendations
                            if result['recommendations']:
                                display_recommendations(result['recommendations'])
                            
                            # Offer pose visualization
                            if st.button("🎨 Show Pose Detection", use_container_width=True):
                                temp_path = save_uploaded_file(uploaded_file)
                                try:
                                    pose_image = classifier.visualize_pose(temp_path)
                                    if pose_image is not None:
                                        st.image(pose_image, caption="Pose Detection Visualization", use_container_width=True)
                                    os.unlink(temp_path)
                                except Exception as e:
                                    st.error(f"Error creating pose visualization: {e}")
                        
                        else:
                            st.markdown(f"""
                            <div class="error-message">
                                <strong>Analysis Failed</strong><br>
                                {result['error']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 💡 Tips for Better Results:")
                            st.markdown("""
                            - Ensure your full body is visible in the image
                            - Stand straight facing the camera
                            - Use good lighting
                            - Avoid baggy clothing that hides your body shape
                            - Make sure the background is not cluttered
                            """)
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
    
    else:
        # Sample images section
        st.markdown("## 🖼️ Sample Images")
        st.markdown("Don't have a photo ready? Here are some tips for taking the perfect body type analysis photo:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### ✅ Good Photo")
            st.markdown("- Full body visible")
            st.markdown("- Standing straight")
            st.markdown("- Good lighting")
            st.markdown("- Minimal background")
        
        with col2:
            st.markdown("### ❌ Avoid")
            st.markdown("- Partial body shots")
            st.markdown("- Sitting or lying down")
            st.markdown("- Dark or blurry images")
            st.markdown("- Baggy clothing")
        
        with col3:
            st.markdown("### 💡 Pro Tips")
            st.markdown("- Use a timer or ask someone to help")
            st.markdown("- Stand 6-8 feet from camera")
            st.markdown("- Wear fitted clothing")
            st.markdown("- Keep arms slightly away from body")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Made with ❤️ using Streamlit and MediaPipe</p>
        <p><em>Your privacy is important - uploaded images are processed locally and not stored</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
