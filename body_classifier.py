import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt

class BodyTypeClassifier:
    def __init__(self):
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Body type definitions and recommendations
        self.body_types = {
            'Rectangle': {
                'description': 'Balanced proportions with similar shoulder, waist, and hip measurements',
                'tops': ['Peplum tops', 'Wrap tops', 'Fitted blazers', 'Cropped jackets'],
                'bottoms': ['High-waisted jeans', 'A-line skirts', 'Wide-leg pants', 'Bootcut jeans'],
                'dresses': ['Wrap dresses', 'Fit-and-flare dresses', 'Belted dresses', 'Sheath dresses'],
                'avoid': ['Straight-cut tops', 'Boxy clothing', 'Drop-waist dresses']
            },
            'Pear': {
                'description': 'Hips wider than shoulders, well-defined waist',
                'tops': ['Boat necks', 'Off-shoulder tops', 'Bright colored tops', 'Statement sleeves'],
                'bottoms': ['Dark colored bottoms', 'Straight-leg jeans', 'Bootcut pants', 'A-line skirts'],
                'dresses': ['A-line dresses', 'Fit-and-flare dresses', 'Empire waist dresses'],
                'avoid': ['Skinny jeans', 'Tight bottoms', 'Hip details', 'Light colored bottoms']
            },
            'Apple': {
                'description': 'Fuller midsection with less defined waist',
                'tops': ['V-neck tops', 'Empire waist tops', 'Tunic tops', 'Wrap tops'],
                'bottoms': ['Straight-leg pants', 'Bootcut jeans', 'A-line skirts', 'High-waisted bottoms'],
                'dresses': ['Empire waist dresses', 'A-line dresses', 'Wrap dresses', 'V-neck dresses'],
                'avoid': ['Tight tops', 'Belts at natural waist', 'Clingy fabrics', 'Crop tops']
            },
            'Hourglass': {
                'description': 'Balanced bust and hips with a well-defined waist',
                'tops': ['Fitted tops', 'Wrap tops', 'V-neck tops', 'Scoop necks'],
                'bottoms': ['High-waisted jeans', 'Pencil skirts', 'Fitted pants', 'Bootcut jeans'],
                'dresses': ['Wrap dresses', 'Bodycon dresses', 'Fit-and-flare dresses', 'Belted dresses'],
                'avoid': ['Oversized clothing', 'Loose fits', 'Low-rise pants', 'Boxy tops']
            },
            'Inverted Triangle': {
                'description': 'Broader shoulders than hips',
                'tops': ['V-neck tops', 'Scoop necks', 'Soft fabrics', 'Darker colored tops'],
                'bottoms': ['Wide-leg pants', 'Flared jeans', 'Bright colored bottoms', 'Detailed bottoms'],
                'dresses': ['A-line dresses', 'Fit-and-flare dresses', 'Peplum dresses'],
                'avoid': ['Shoulder pads', 'Boat necks', 'Horizontal stripes on top', 'Strapless tops']
            }
        }
    
    def extract_landmarks(self, image_path):
        """Extract pose landmarks from image using MediaPipe"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, image_rgb
        
        return results.pose_landmarks, image_rgb
    
    def calculate_body_measurements(self, landmarks):
        """Calculate body measurements from pose landmarks"""
        # Get landmark coordinates
        lm = landmarks.landmark
        
        # Key landmarks
        left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate measurements
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        
        # Estimate waist (midpoint between shoulder and hip)
        waist_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        waist_width = shoulder_width * 0.75  # Estimate waist as 75% of shoulder width
        
        # Calculate torso length
        torso_length = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
        
        return {
            'shoulder_width': shoulder_width,
            'waist_width': waist_width,
            'hip_width': hip_width,
            'torso_length': torso_length
        }
    
    def classify_body_type(self, measurements):
        """Classify body type based on measurements"""
        shoulder = measurements['shoulder_width']
        waist = measurements['waist_width']
        hip = measurements['hip_width']
        
        # Calculate ratios
        shoulder_hip_ratio = shoulder / hip if hip > 0 else 1
        waist_hip_ratio = waist / hip if hip > 0 else 1
        shoulder_waist_ratio = shoulder / waist if waist > 0 else 1
        
        # Classification logic with confidence scores
        confidence_scores = {}
        
        # Rectangle: Similar measurements all around
        if 0.85 <= shoulder_hip_ratio <= 1.15 and waist_hip_ratio > 0.75:
            confidence_scores['Rectangle'] = min(1.0, 1.0 - abs(shoulder_hip_ratio - 1.0) * 2)
        
        # Pear: Hips wider than shoulders
        if shoulder_hip_ratio < 0.85:
            confidence_scores['Pear'] = min(1.0, (0.85 - shoulder_hip_ratio) * 2)
        
        # Inverted Triangle: Shoulders wider than hips
        if shoulder_hip_ratio > 1.15:
            confidence_scores['Inverted Triangle'] = min(1.0, (shoulder_hip_ratio - 1.15) * 2)
        
        # Apple: Larger waist compared to hips and shoulders
        if waist_hip_ratio > 0.85 and shoulder_hip_ratio > 0.9:
            confidence_scores['Apple'] = min(1.0, waist_hip_ratio)
        
        # Hourglass: Balanced shoulders and hips with smaller waist
        if 0.9 <= shoulder_hip_ratio <= 1.1 and waist_hip_ratio < 0.75:
            confidence_scores['Hourglass'] = min(1.0, 1.0 - waist_hip_ratio)
        
        # If no clear classification, default to Rectangle
        if not confidence_scores:
            confidence_scores['Rectangle'] = 0.5
        
        # Get the body type with highest confidence
        body_type = max(confidence_scores, key=confidence_scores.get)
        confidence = confidence_scores[body_type]
        
        return body_type, confidence, confidence_scores
    
    def analyze_image(self, image_path):
        """Main function to analyze image and return body type classification"""
        try:
            # Extract landmarks
            landmarks, image = self.extract_landmarks(image_path)
            
            if landmarks is None:
                return {
                    'success': False,
                    'error': 'No pose detected in image. Please use a clear full-body image.',
                    'image': image
                }
            
            # Calculate measurements
            measurements = self.calculate_body_measurements(landmarks)
            
            # Classify body type
            body_type, confidence, all_scores = self.classify_body_type(measurements)
            
            # Get recommendations
            recommendations = self.body_types[body_type]
            
            return {
                'success': True,
                'body_type': body_type,
                'confidence': confidence,
                'all_scores': all_scores,
                'measurements': measurements,
                'recommendations': recommendations,
                'landmarks': landmarks,
                'image': image
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image': None
            }
    
    def visualize_results(self, result, save_path=None):
        """Visualize the analysis results"""
        if not result['success']:
            print(f"Analysis failed: {result['error']}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Original image with pose landmarks
        axes[0].imshow(result['image'])
        axes[0].set_title('Pose Detection')
        axes[0].axis('off')
        
        # Draw pose landmarks if available
        if result['landmarks']:
            # Convert landmarks to pixel coordinates
            h, w = result['image'].shape[:2]
            landmarks = result['landmarks']
            
            # Draw key points
            key_points = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ]
            
            for point in key_points:
                landmark = landmarks.landmark[point]
                x, y = int(landmark.x * w), int(landmark.y * h)
                axes[0].plot(x, y, 'ro', markersize=8)
        
        # Plot 2: Body type classification results
        body_types = list(result['all_scores'].keys())
        scores = list(result['all_scores'].values())
        colors = ['#FF6B6B' if bt == result['body_type'] else '#4ECDC4' for bt in body_types]
        
        bars = axes[1].bar(body_types, scores, color=colors)
        axes[1].set_title(f'Body Type Classification\nPredicted: {result["body_type"]} (Confidence: {result["confidence"]:.2f})')
        axes[1].set_ylabel('Confidence Score')
        axes[1].set_ylim(0, 1)
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_recommendations(self, result):
        """Print styling recommendations"""
        if not result['success']:
            print(f"Cannot provide recommendations: {result['error']}")
            return
        
        body_type = result['body_type']
        recommendations = result['recommendations']
        confidence = result['confidence']
        
        print(f"\n{'='*60}")
        print(f"BODY TYPE ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Predicted Body Type: {body_type}")
        print(f"Confidence Score: {confidence:.2%}")
        print(f"Description: {recommendations['description']}")
        
        print(f"\n{'='*60}")
        print(f"STYLING RECOMMENDATIONS")
        print(f"{'='*60}")
        
        print(f"\n👔 TOPS:")
        for item in recommendations['tops']:
            print(f"  ✓ {item}")
        
        print(f"\n👖 BOTTOMS:")
        for item in recommendations['bottoms']:
            print(f"  ✓ {item}")
        
        print(f"\n👗 DRESSES:")
        for item in recommendations['dresses']:
            print(f"  ✓ {item}")
        
        print(f"\n❌ AVOID:")
        for item in recommendations['avoid']:
            print(f"  ✗ {item}")
        
        print(f"\n{'='*60}")
        
        # Print all confidence scores
        print(f"All Body Type Scores:")
        for bt, score in result['all_scores'].items():
            print(f"  {bt}: {score:.2%}")
        
        print(f"{'='*60}\n")

    # Compatibility methods for Streamlit app
    def analyze_body_type(self, image_path):
        """Streamlit-compatible analysis method"""
        result = self.analyze_image(image_path)
        
        if not result['success']:
            return {
                'success': False,
                'error': result['error']
            }
        
        # Convert to Streamlit-expected format
        measurements = {
            'shoulder_hip_ratio': result['measurements']['shoulder_width'] / result['measurements']['hip_width'],
            'waist_hip_ratio': result['measurements']['waist_width'] / result['measurements']['hip_width'], 
            'waist_definition': 1.0 - result['measurements']['waist_width'] / result['measurements']['shoulder_width'],
            'torso_length': result['measurements']['torso_length']
        }
        
        return {
            'success': True,
            'body_type': result['body_type'],
            'confidence': result['confidence'],
            'measurements': measurements,
            'recommendations': result['recommendations']
        }
    
    def visualize_pose(self, image_path):
        """Create pose visualization for Streamlit"""
        try:
            landmarks, image = self.extract_landmarks(image_path)
            if landmarks is None:
                return None
            
            # Draw pose landmarks on image
            annotated_image = image.copy()
            
            # Convert landmarks to pixel coordinates
            h, w = image.shape[:2]
            
            # Draw key points
            key_points = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ]
            
            # Draw circles on key landmarks
            for point in key_points:
                landmark = landmarks.landmark[point]
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Draw circle on the image
                import cv2
                cv2.circle(annotated_image, (x, y), 8, (255, 0, 0), -1)
            
            return annotated_image
            
        except Exception as e:
            print(f"Error creating pose visualization: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = BodyTypeClassifier()
    
    # Example analysis
    image_path = "path_to_your_image.jpg"  # Replace with actual image path
    
    # Analyze image
    result = classifier.analyze_image(image_path)
    
    # Print recommendations
    classifier.print_recommendations(result)
    
    # Visualize results
    classifier.visualize_results(result)
