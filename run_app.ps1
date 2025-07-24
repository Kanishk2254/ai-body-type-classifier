# Body Type Classifier - Run with suppressed warnings
# PowerShell script to set environment variables and run the app

Write-Host "Starting Body Type Classifier..." -ForegroundColor Green

# TensorFlow warning suppressions
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# MediaPipe suppressions
$env:MEDIAPIPE_DISABLE_GPU = "1"
$env:GLOG_minloglevel = "3"
$env:GLOG_v = "0"

# Protobuf suppressions
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Python warnings
$env:PYTHONWARNINGS = "ignore"

Write-Host "Environment configured. Launching Streamlit app..." -ForegroundColor Yellow

# Run the Streamlit app
try {
    streamlit run app.py
} catch {
    Write-Host "Error running the app: $_" -ForegroundColor Red
    Write-Host "Make sure you have installed all dependencies:" -ForegroundColor Yellow
    Write-Host "pip install -r requirements.txt" -ForegroundColor Cyan
}

Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
