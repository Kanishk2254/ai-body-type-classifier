# AI Body Type Analyzer

A sleek AI-powered web application that analyzes body types from uploaded images and provides personalized clothing recommendations using MediaPipe pose estimation.

## 🌟 Features

- **AI-Powered Body Type Classification**: Uses MediaPipe pose estimation to analyze body proportions
- **5 Body Type Categories**: 
  - 🍎 Apple (Round)
  - 🍐 Pear (Triangle)
  - ⏳ Hourglass
  - 📏 Rectangle (I)
  - 🔺 Inverted Triangle
- **Personalized Style Recommendations**: Tailored clothing suggestions for each body type
- **Web Interface**: User-friendly Streamlit app for easy image upload and analysis
- **Command-Line Interface**: CLI tool for batch processing and testing
- **Confidence Scoring**: Reliability assessment for each prediction

## 🚀 Quick Start

### Installation

1. **Clone or download the project**
```bash
cd body-classfier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Web Interface
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

#### Python API
```python
from body_classifier import BodyTypeClassifier

classifier = BodyTypeClassifier()
result = classifier.analyze_body_type("your_image.jpg")
print(result)
```

## 📸 Image Guidelines

For best results, use images with:
- **Full body visible** from head to feet
- **Clear pose** with arms at sides
- **Fitted clothing** to show body silhouette
- **Good lighting** and minimal background distractions
- **Front-facing** or slightly angled pose

### 🖼️ Sample Images
Check the `images/` folder for:
- Sample test images
- Body type examples
- App screenshots
- Usage guidelines

See `images/README.md` for detailed information about image requirements and examples.

## 🧠 How It Works

### 1. Pose Estimation
- Uses Google's MediaPipe to extract 33 body landmarks
- Identifies key points: shoulders, waist, hips

### 2. Body Measurement Calculation
- Computes shoulder width, hip width, and estimated waist width
- Calculates ratios: shoulder-to-hip, waist-to-shoulder, waist-to-hip

### 3. Classification Algorithm
```python
# Simplified classification logic
if shoulder_to_hip_ratio > 1.15:
    return "Inverted Triangle"
elif shoulder_to_hip_ratio < 0.85:
    return "Pear (Triangle)"
elif waist_is_narrow and shoulders_similar_to_hips:
    return "Hourglass"
elif measurements_are_similar:
    return "Rectangle (I)"
else:
    return "Apple (Round)"
```

### 4. Style Recommendations
- Rule-based recommendation system
- Curated suggestions for tops, bottoms, dresses, accessories
- Professional styling tips for each body type

## 📊 Body Type Characteristics

### Rectangle (I) Body Type
- **Characteristics**: Shoulders ≈ Waist ≈ Hips
- **Goals**: Create curves, define waist
- **Recommendations**: Peplum tops, A-line skirts, wide belts

### Hourglass Body Type
- **Characteristics**: Bust ≈ Hips, narrow waist
- **Goals**: Emphasize natural waistline
- **Recommendations**: Fitted tops, pencil skirts, wrap dresses

### Inverted Triangle Body Type
- **Characteristics**: Broad shoulders, narrow hips
- **Goals**: Balance proportions, soften shoulders
- **Recommendations**: V-necks, wide-leg pants, A-line dresses

### Pear (Triangle) Body Type
- **Characteristics**: Hips wider than shoulders
- **Goals**: Balance proportions, emphasize upper body
- **Recommendations**: Boat necks, structured blazers, dark bottoms

### Apple (Round) Body Type
- **Characteristics**: Fuller midsection, less defined waist
- **Goals**: Create vertical lines, emphasize legs
- **Recommendations**: Empire waist, flowy tunics, straight-leg pants

## 🛠️ Technical Details

### Dependencies
- **OpenCV**: Image processing
- **MediaPipe**: Pose estimation and landmark detection
- **Streamlit**: Web interface
- **NumPy**: Numerical computations
- **PIL/Pillow**: Image handling
- **Matplotlib**: Visualization (optional)

### Architecture
```
body-classfier/
├── body_classifier.py    # Core classification logic
├── app.py               # Streamlit web interface
├── requirements.txt     # Dependencies
├── images/              # Sample images and screenshots
│   ├── README.md        # Image guidelines and documentation
│   └── .gitkeep         # Ensures folder is tracked by Git
└── README.md           # This file
```

### Key Classes
- **BodyTypeClassifier**: Main classifier with pose estimation
- **Methods**:
  - `extract_pose_landmarks()`: MediaPipe pose detection
  - `calculate_body_measurements()`: Ratio calculations
  - `classify_body_type()`: Classification logic
  - `get_clothing_recommendations()`: Style suggestions

## 🔧 Customization

### Adjust Classification Thresholds
Modify the thresholds in `classify_body_type()` method:
```python
similar_threshold = 0.95  # Within 5% considered similar
waist_threshold = 0.75    # Waist significantly smaller
```

### Add New Recommendations
Extend the recommendations dictionary in `get_clothing_recommendations()`:
```python
recommendations = {
    "Rectangle (I)": {
        "tops": ["Your new suggestions here"],
        # ... more categories
    }
}
```

### Improve Classification
- Collect more training data
- Implement machine learning models
- Add additional body landmarks
- Fine-tune measurement calculations

## 🎯 Future Enhancements

- [ ] **Machine Learning Model**: Train on larger datasets
- [ ] **More Body Types**: Add additional categories
- [ ] **Color Analysis**: Seasonal color recommendations
- [ ] **Size Recommendations**: Suggest optimal clothing sizes
- [ ] **Shopping Integration**: Link to online stores
- [ ] **Mobile App**: React Native or Flutter app
- [ ] **3D Analysis**: Depth-based measurements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## ⚠️ Disclaimer

This tool provides style suggestions based on general body type guidelines. Fashion is subjective, and personal preferences should always take precedence. The AI predictions are estimates and may not be 100% accurate for all body types or poses.

## 🆘 Troubleshooting

### Common Issues

1. **"Could not detect pose landmarks"**
   - Ensure full body is visible in image
   - Check image quality and lighting
   - Try different poses or angles

2. **MediaPipe installation issues**
   - Update pip: `pip install --upgrade pip`
   - Install MediaPipe separately: `pip install mediapipe`

3. **Streamlit not loading**
   - Check if port 8501 is available
   - Try: `streamlit run app.py --port 8502`

### Screenshots

The app features a modern, intuitive interface with:
- Beautiful gradient headers
- Interactive confidence bars
- Organized recommendation tabs
- Pose visualization capability

### Getting Help
- Check the error messages for specific guidance
- Ensure all dependencies are installed correctly
- Verify image file formats (JPG, PNG supported)

---

**Built with ❤️ using Python, MediaPipe, and Streamlit**
