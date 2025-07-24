@echo off
REM Body Type Classifier - Run with suppressed warnings
REM Set environment variables to suppress TensorFlow and other warnings

echo Starting Body Type Classifier...

REM TensorFlow warning suppressions
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set CUDA_VISIBLE_DEVICES=-1
set TF_FORCE_GPU_ALLOW_GROWTH=true

REM MediaPipe suppressions
set MEDIAPIPE_DISABLE_GPU=1
set GLOG_minloglevel=3
set GLOG_v=0

REM Protobuf suppressions
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Python warnings
set PYTHONWARNINGS=ignore

echo Environment configured. Launching Streamlit app...
streamlit run app.py

pause
