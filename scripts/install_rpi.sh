#!/bin/bash
# Installs the dependencies on Raspberry Pi OS and starts the app.
# The apt packages are prebuilt for the Pi's ARM CPU, which avoids compiling OpenCV
# (that takes hours on a Pi 3 B+ with 1 GB of RAM).
set -e

cd "$(dirname "$0")/.."

sudo apt update
sudo apt install -y python3-opencv python3-flask python3-sklearn python3-pandas python3-joblib

# the webcam must show up as /dev/video0
if [ ! -e /dev/video0 ]; then
    echo "No camera found on /dev/video0, check the USB webcam."
    exit 1
fi

python3 app.py
