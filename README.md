# 🖥️ Monitor Area of Object – YOLO Real-Time Inference GUI

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-yolov8](https://img.shields.io/badge/Made%20with-YOLOv8-00BFFF.svg)](https://github.com/ultralytics/ultralytics)

> A **real-time object area monitoring** application built with **Python**, **YOLOv8**, and **Tkinter**.  
> Provides video stream detection, area trend analysis with SMA smoothing, and automatic alerts with PDF reports.

![GUI Screenshot](https://github.com/mai890107/monitor-area-of-object-GUI/raw/main/docs/gui%20screenshot.jpg)

---

## ✨ Features

- 🎬 **Multiple Input Sources**: Load from local video files, RTSP streams, or camera devices.  
- 🤖 **YOLOv8 Inference**: GPU-accelerated object detection using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).  
- 📊 **Area Monitoring**: Computes object areas per frame, applies **SMA smoothing**, and plots real-time trends.  
- ⚠️ **Alerts & Reports**: NG (No Good) alerts trigger sound notifications and **auto-generated PDF reports** (with start/NG images).  
- 🖼️ **Interactive GUI**: Adjustable confidence threshold, FPS, SMA window, and more, all in a modern Tkinter interface.  
- 💾 **Video Export**: Option to save annotated output videos.  

---

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/mai890107/monitor-area-of-object-GUI.git
cd monitor-area-of-object-GUI
2️⃣ Create a Virtual Environment (Recommended)
macOS / Linux
python -m venv venv
source venv/bin/activate
Windows
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Usage
python app.py

Steps

Load YOLO Model → Choose area0903.pt or a pretrained yolov8*.pt model.
Select Video Source → Upload a video, open a local camera, or provide an RTSP URL.
Adjust Parameters → Set confidence, FPS, SMA window, and output saving options.
Start Inference → Click ▶ to begin detection and area trend monitoring.
Monitor NG Alerts → Receive audible beeps and auto-generated PDF reports when NG conditions are met.

This project is licensed under the MIT License
.





