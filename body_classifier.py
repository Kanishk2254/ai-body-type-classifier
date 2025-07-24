# Import config first to suppress warnings
import config

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional
import math
from PIL import Image


class BodyTypeClassifier:
    """
    A class to classify body types from images using MediaPipe pose estimation.
    
    Body Types:
    - Rectangle (I): Shoulders ≈ Waist ≈ Hips
    - Inverted Triangle: Broad shoulders, narrow waist/hips
    - Pear (Triangle): Hips wider than shoulders
    - Hourglass: Bust ≈ Hips, narrow waist
    - Apple (Round): Broad torso, undefined waist
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_pose_landmarks(self, image_path: str) -> Optional[Dict]:
        """Extract pose landmarks from an image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return None
                
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = self.mp_pose.PoseLandmark(idx).name
                landmarks[landmark_name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            return landmarks
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def calculate_body_measurements(self, landmarks: Dict) -> Optional[Dict]:
        """Calculate body measurements from pose landmarks."""
        try:
            # Key landmarks
            left_shoulder = landmarks['LEFT_SHOULDER']
            right_shoulder = landmarks['RIGHT_SHOULDER']
            left_hip = landmarks['LEFT_HIP']
            right_hip = landmarks['RIGHT_HIP']
            
            # Calculate widths
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            hip_width = abs(left_hip['x'] - right_hip['x'])
            
            # Estimate waist width (approximation between shoulders and hips)
            waist_width = (shoulder_width + hip_width) / 2.2  # Empirical adjustment
            
            # Calculate torso length
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            torso_length = abs(hip_center_y - shoulder_center_y)
            
            # Calculate ratios
            shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 1
            waist_to_shoulder_ratio = waist_width / shoulder_width if shoulder_width > 0 else 1
            waist_to_hip_ratio = waist_width / hip_width if hip_width > 0 else 1
            
            measurements = {
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'waist_width': waist_width,
                'torso_length': torso_length,
                'shoulder_to_hip_ratio': shoulder_to_hip_ratio,
                'waist_to_shoulder_ratio': waist_to_shoulder_ratio,
                'waist_to_hip_ratio': waist_to_hip_ratio
            }
            
            return measurements
            
        except KeyError as e:
            print(f"Missing landmark: {e}")
            return None
        except Exception as e:
            print(f"Error calculating measurements: {e}")
            return None
    
    def classify_body_type(self, measurements: Dict) -> str:
        """Classify body type based on measurements."""
        shoulder_to_hip = measurements['shoulder_to_hip_ratio']
        waist_to_shoulder = measurements['waist_to_shoulder_ratio']
        waist_to_hip = measurements['waist_to_hip_ratio']
        
        # Classification thresholds (can be fine-tuned)
        similar_threshold = 0.95  # Within 5% considered similar
        waist_threshold = 0.75    # Waist significantly smaller
        
        # Hourglass: Similar shoulders and hips, narrow waist
        if (abs(shoulder_to_hip - 1.0) < (1 - similar_threshold) and 
            waist_to_shoulder < waist_threshold and 
            waist_to_hip < waist_threshold):
            return "Hourglass"
        
        # Rectangle (I): Similar measurements across shoulders, waist, hips
        elif (abs(shoulder_to_hip - 1.0) < (1 - similar_threshold) and
              waist_to_shoulder > waist_threshold and
              waist_to_hip > waist_threshold):
            return "Rectangle (I)"
        
        # Inverted Triangle: Shoulders significantly wider than hips
        elif shoulder_to_hip > 1.15:
            return "Inverted Triangle"
        
        # Pear (Triangle): Hips significantly wider than shoulders
        elif shoulder_to_hip < 0.85:
            return "Pear (Triangle)"
        
        # Apple: Undefined waist, broader torso
        else:
            return "Apple (Round)"
    
    def get_clothing_recommendations(self, body_type: str) -> Dict:
        """Get clothing recommendations based on body type."""
        recommendations = {
            "Rectangle (I)": {
                "tops": ["Peplum tops", "Ruffled blouses", "Wrap tops", "Belted jackets"],
                "bottoms": ["A-line skirts", "Flared pants", "Layered skirts"],
                "dresses": ["Fit-and-flare dresses", "Wrap dresses", "Empire waist dresses"],
                "accessories": ["Wide belts", "Statement jewelry", "Scarves"],
                "tips": ["Create curves with layers", "Define waist with belts", "Add volume to bust and hips"]
            },
            "Hourglass": {
                "tops": ["Fitted tops", "V-neck blouses", "Wrap tops", "Button-down shirts"],
                "bottoms": ["High-waist trousers", "Pencil skirts", "Fitted jeans"],
                "dresses": ["Bodycon dresses", "Wrap dresses", "Sheath dresses"],
                "accessories": ["Thin belts", "Classic jewelry", "Structured bags"],
                "tips": ["Emphasize your waist", "Choose fitted silhouettes", "Avoid oversized clothing"]
            },
            "Inverted Triangle": {
                "tops": ["V-neck tops", "Scoop necks", "Soft fabrics", "Dark colors on top"],
                "bottoms": ["Wide-leg pants", "A-line skirts", "Bright colored bottoms"],
                "dresses": ["A-line dresses", "Empire waist dresses", "Maxi dresses"],
                "accessories": ["Hip belts", "Statement necklaces", "Bright shoes"],
                "tips": ["Balance broad shoulders", "Add volume to lower body", "Draw attention downward"]
            },
            "Pear (Triangle)": {
                "tops": ["Boat necks", "Off-shoulder tops", "Bright colored tops", "Structured blazers"],
                "bottoms": ["Straight-leg pants", "Dark colored bottoms", "High-waist jeans"],
                "dresses": ["A-line dresses", "Fit-and-flare dresses", "Empire waist"],
                "accessories": ["Statement earrings", "Colorful scarves", "Structured shoulder bags"],
                "tips": ["Emphasize shoulders and bust", "Choose dark bottoms", "Add structure to upper body"]
            },
            "Apple (Round)": {
                "tops": ["Empire waist tops", "V-neck blouses", "Flowy tunics", "Open cardigans"],
                "bottoms": ["Straight-leg pants", "Bootcut jeans", "A-line skirts"],
                "dresses": ["Empire waist dresses", "A-line dresses", "Wrap dresses"],
                "accessories": ["Long necklaces", "Vertical patterns", "Structured bags"],
                "tips": ["Create vertical lines", "Emphasize legs", "Choose flowy fabrics around torso"]
            }
        }
        
        return recommendations.get(body_type, {})
    
    def analyze_image(self, image_path: str) -> Dict:
        """Complete analysis pipeline for a single image."""
        # Extract landmarks
        landmarks = self.extract_pose_landmarks(image_path)
        if not landmarks:
            return {"error": "Could not detect pose landmarks in the image"}
        
        # Calculate measurements
        measurements = self.calculate_body_measurements(landmarks)
        if not measurements:
            return {"error": "Could not calculate body measurements"}
        
        # Classify body type
        body_type = self.classify_body_type(measurements)
        
        # Get recommendations
        recommendations = self.get_clothing_recommendations(body_type)
        
        return {
            "body_type": body_type,
            "measurements": measurements,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(measurements)
        }
    
    def _calculate_confidence(self, measurements: Dict) -> float:
        """Calculate confidence score based on measurement quality."""
        # Simple confidence calculation based on measurement ratios
        # Higher confidence for more distinct body types
        shoulder_to_hip = measurements['shoulder_to_hip_ratio']
        
        if abs(shoulder_to_hip - 1.0) > 0.15:  # Very distinct ratios
            return 0.9
        elif abs(shoulder_to_hip - 1.0) > 0.10:  # Moderately distinct
            return 0.75
        else:  # Less distinct
            return 0.6


if __name__ == "__main__":
    # Example usage
    classifier = BodyTypeClassifier()
    
    # Test with an image (you'll need to provide a valid image path)
    # result = classifier.analyze_image("path_to_your_image.jpg")
    # print(result)
    
    print("Body Type Classifier initialized successfully!")
    print("Use classifier.analyze_image('path_to_image.jpg') to analyze an image.")
