#!/bin/bash

# Baseball Impact Detection System - Setup & Test Script
# Run this on your Raspberry Pi 5 to install and test the system

set -e  # Exit on error

echo "========================================"
echo "Baseball Impact Detection Setup"
echo "========================================"
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "WARNING: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update package list
echo "1. Updating package list..."
sudo apt-get update

# Install system dependencies
echo ""
echo "2. Installing system dependencies..."
sudo apt-get install -y python3-opencv python3-picamera2 python3-numpy python3-pip

# Install Python packages
echo ""
echo "3. Installing Python packages..."
pip3 install --break-system-packages numpy opencv-python

# Verify camera
echo ""
echo "4. Checking camera..."
if command -v libcamera-hello &> /dev/null; then
    echo "Testing camera with libcamera-hello..."
    timeout 3s libcamera-hello --list-cameras || true
    echo "Camera check complete"
else
    echo "WARNING: libcamera-hello not found"
    echo "Camera may not be properly configured"
fi

# Set CPU to performance mode
echo ""
echo "5. Setting CPU to performance mode..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo "CPU governor set to performance"

# Create test directory
echo ""
echo "6. Setting up test environment..."
mkdir -p ~/baseball_detector_logs
cd ~/baseball_detector_logs

# Copy or verify main script exists
if [ ! -f "../baseball_impact_detector.py" ]; then
    echo "ERROR: baseball_impact_detector.py not found in parent directory"
    echo "Please ensure the script is in the same directory as this setup script"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Quick Test Commands:"
echo ""
echo "1. Test camera only (5 seconds):"
echo "   libcamera-hello -t 5000"
echo ""
echo "2. Run detector with calibration:"
echo "   python3 ../baseball_impact_detector.py"
echo ""
echo "3. Run detector without calibration (quick test):"
echo "   python3 ../baseball_impact_detector.py --no-calibration"
echo ""
echo "4. Run detector headless (max performance):"
echo "   python3 ../baseball_impact_detector.py --no-calibration --no-display"
echo ""
echo "5. Monitor system performance:"
echo "   htop  # Install with: sudo apt-get install htop"
echo ""
echo "Logs will be saved in: ~/baseball_detector_logs/"
echo ""
echo "For troubleshooting, see PERFORMANCE_GUIDE.md"
echo ""

# Optional: Run a quick test
read -p "Run a quick test now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting detector in test mode (press 'q' to quit)..."
    echo "Wave your hand in front of the camera to test detection"
    sleep 2
    cd ..
    python3 baseball_impact_detector.py --no-calibration || echo "Test completed or interrupted"
fi

echo ""
echo "Setup script finished!"
