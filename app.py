# Import config first to suppress warnings
import config

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from body_classifier import BodyTypeClassifier
from image_utils import process_uploaded_image, ImageProcessor
import matplotlib.pyplot as plt


def main():
    st.set_page_config(
        page_title="Body Type Classifier",
        page_icon="👗",
        layout="wide"
    )
    
    st.title("🎯 AI Body Type Classifier & Style Advisor")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This AI-powered app analyzes your body type from a full-body image and provides personalized clothing recommendations.
        
        **Supported Body Types:**
        - 🍎 Apple (Round)
        - 🍐 Pear (Triangle) 
        - ⏳ Hourglass
        - 📏 Rectangle (I)
        - 🔺 Inverted Triangle
        """)
        
        st.header("📸 Image Guidelines")
        st.markdown("""
        For best results:
        - Use a full-body image
        - Stand straight with arms at sides
        - Wear fitted clothing
        - Good lighting and clear background
        - Face the camera directly
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a full-body image for analysis"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image using utility function
            temp_path = process_uploaded_image(uploaded_file)
    
    with col2:
        if uploaded_file is not None:
            st.header("Analysis Results")
            
            # Check if image processing was successful
            if temp_path is None:
                st.error("❌ Failed to process the uploaded image")
                st.info("💡 Please try uploading a different image format (JPG, PNG)")
                return
            
            # Initialize classifier and analyze
            with st.spinner("🔍 Analyzing your body type..."):
                try:
                    classifier = BodyTypeClassifier()
                    result = classifier.analyze_image(temp_path)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        st.info("💡 Try uploading a clearer full-body image with better pose visibility.")
                    else:
                        # Display results
                        display_results(result)
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass


def display_results(result):
    """Display the analysis results in an organized manner."""
    body_type = result['body_type']
    confidence = result['confidence']
    measurements = result['measurements']
    recommendations = result['recommendations']
    
    # Body type and confidence
    st.success(f"🎯 **Detected Body Type:** {body_type}")
    
    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
    st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence:.1%}</span>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Measurements visualization
    with st.expander("📏 Body Measurements Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Shoulder to Hip Ratio", f"{measurements['shoulder_to_hip_ratio']:.2f}")
            st.metric("Waist to Shoulder Ratio", f"{measurements['waist_to_shoulder_ratio']:.2f}")
        
        with col2:
            st.metric("Waist to Hip Ratio", f"{measurements['waist_to_hip_ratio']:.2f}")
            st.metric("Torso Length", f"{measurements['torso_length']:.3f}")
    
    # Clothing recommendations
    if recommendations:
        st.header("👗 Personalized Style Recommendations")
        
        # Create tabs for different categories
        tabs = st.tabs(["👚 Tops", "👖 Bottoms", "👗 Dresses", "✨ Accessories", "💡 Style Tips"])
        
        with tabs[0]:  # Tops
            st.subheader("Recommended Tops")
            for item in recommendations.get('tops', []):
                st.write(f"• {item}")
        
        with tabs[1]:  # Bottoms
            st.subheader("Recommended Bottoms")
            for item in recommendations.get('bottoms', []):
                st.write(f"• {item}")
        
        with tabs[2]:  # Dresses
            st.subheader("Recommended Dresses")
            for item in recommendations.get('dresses', []):
                st.write(f"• {item}")
        
        with tabs[3]:  # Accessories
            st.subheader("Recommended Accessories")
            for item in recommendations.get('accessories', []):
                st.write(f"• {item}")
        
        with tabs[4]:  # Tips
            st.subheader("Style Tips")
            for tip in recommendations.get('tips', []):
                st.info(f"💡 {tip}")
    
    # Body type description
    st.markdown("---")
    st.header("📖 About Your Body Type")
    body_type_info = get_body_type_description(body_type)
    st.markdown(body_type_info)


def get_body_type_description(body_type):
    """Get detailed description for each body type."""
    descriptions = {
        "Rectangle (I)": """
        **Rectangle (I) Body Type:**
        Your shoulders, waist, and hips are similar in width, creating a straight silhouette. 
        This body type is also known as the "athletic" or "straight" body type.
        
        **Key Characteristics:**
        - Minimal waist definition
        - Similar shoulder and hip measurements
        - Weight tends to be evenly distributed
        
        **Styling Goals:**
        - Create the illusion of curves
        - Define your waistline
        - Add volume to bust and hips
        """,
        
        "Hourglass": """
        **Hourglass Body Type:**
        You have a well-defined waist with bust and hip measurements that are nearly equal. 
        This is considered the classic feminine silhouette.
        
        **Key Characteristics:**
        - Defined waistline
        - Balanced bust and hip proportions
        - Curves in all the right places
        
        **Styling Goals:**
        - Emphasize your natural waistline
        - Choose fitted silhouettes
        - Maintain your balanced proportions
        """,
        
        "Inverted Triangle": """
        **Inverted Triangle Body Type:**
        Your shoulders are broader than your hips, creating a strong upper body silhouette. 
        This body type is common among athletes.
        
        **Key Characteristics:**
        - Broad shoulders
        - Narrow hips
        - Strong, athletic build
        
        **Styling Goals:**
        - Balance your proportions
        - Add volume to your lower body
        - Soften your shoulder line
        """,
        
        "Pear (Triangle)": """
        **Pear (Triangle) Body Type:**
        Your hips are wider than your shoulders, creating a feminine silhouette with curves 
        concentrated in the lower body.
        
        **Key Characteristics:**
        - Narrow shoulders
        - Fuller hips and thighs
        - Defined waistline
        
        **Styling Goals:**
        - Balance your proportions
        - Emphasize your upper body
        - Create shoulder definition
        """,
        
        "Apple (Round)": """
        **Apple (Round) Body Type:**
        You carry weight around your midsection with a fuller bust and less defined waistline. 
        Your legs are often your best feature.
        
        **Key Characteristics:**
        - Fuller midsection
        - Less defined waistline
        - Beautiful legs
        
        **Styling Goals:**
        - Create vertical lines
        - Emphasize your legs
        - Draw attention away from the midsection
        """
    }
    
    return descriptions.get(body_type, "Body type information not available.")


if __name__ == "__main__":
    main()
