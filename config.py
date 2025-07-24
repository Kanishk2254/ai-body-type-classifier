"""
Configuration settings for the Body Type Classifier.
Handles TensorFlow warnings and optimization settings.
"""

import os
import warnings
import logging

def setup_tensorflow_environment():
    """
    Configure TensorFlow environment variables to suppress warnings and optimize performance.
    """
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR logs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid warnings
    
    # Additional TensorFlow optimizations
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage (disable GPU warnings)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Suppress other warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Configure logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # MediaPipe specific suppressions
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    
    # Protobuf warnings
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def setup_mediapipe_environment():
    """
    Configure MediaPipe to run quietly.
    """
    # Disable MediaPipe logging
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['GLOG_v'] = '0'

def initialize_environment():
    """
    Initialize all environment configurations.
    Call this before importing TensorFlow or MediaPipe.
    """
    setup_tensorflow_environment()
    setup_mediapipe_environment()
    
    # Additional general suppressions
    import sys
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

# Initialize on import
initialize_environment()
