#!/usr/bin/env python3
"""
Command-line interface for testing the Body Type Classifier.
Usage: python cli_test.py <image_path>
"""

# Import config first to suppress warnings
import config

import sys
import os
import json
from body_classifier import BodyTypeClassifier


def print_results(result):
    """Print analysis results in a formatted way."""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("=" * 60)
    print("🎯 BODY TYPE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Detected Body Type: {result['body_type']}")
    print(f"🎯 Confidence: {result['confidence']:.1%}")
    
    print("\n📏 Body Measurements:")
    measurements = result['measurements']
    print(f"  • Shoulder to Hip Ratio: {measurements['shoulder_to_hip_ratio']:.3f}")
    print(f"  • Waist to Shoulder Ratio: {measurements['waist_to_shoulder_ratio']:.3f}")
    print(f"  • Waist to Hip Ratio: {measurements['waist_to_hip_ratio']:.3f}")
    print(f"  • Torso Length: {measurements['torso_length']:.3f}")
    
    recommendations = result['recommendations']
    if recommendations:
        print(f"\n👗 STYLE RECOMMENDATIONS FOR {result['body_type']}:")
        print("-" * 50)
        
        if 'tops' in recommendations:
            print("\n👚 Recommended Tops:")
            for item in recommendations['tops']:
                print(f"  • {item}")
        
        if 'bottoms' in recommendations:
            print("\n👖 Recommended Bottoms:")
            for item in recommendations['bottoms']:
                print(f"  • {item}")
        
        if 'dresses' in recommendations:
            print("\n👗 Recommended Dresses:")
            for item in recommendations['dresses']:
                print(f"  • {item}")
        
        if 'accessories' in recommendations:
            print("\n✨ Recommended Accessories:")
            for item in recommendations['accessories']:
                print(f"  • {item}")
        
        if 'tips' in recommendations:
            print("\n💡 Style Tips:")
            for tip in recommendations['tips']:
                print(f"  • {tip}")
    
    print("\n" + "=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python cli_test.py <image_path>")
        print("Example: python cli_test.py sample_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return
    
    print(f"🔍 Analyzing image: {image_path}")
    print("Please wait...")
    
    try:
        # Initialize classifier
        classifier = BodyTypeClassifier()
        
        # Analyze the image
        result = classifier.analyze_image(image_path)
        
        # Print results
        print_results(result)
        
        # Optionally save results to JSON
        output_file = f"analysis_result_{os.path.basename(image_path)}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
