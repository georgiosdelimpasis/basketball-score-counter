# 🎯 YOLO Real-Time Webcam Detection App

A production-ready Streamlit application for real-time object detection using YOLOv8, OpenCV, and Streamlit. Detect objects from your webcam with adjustable parameters and beautiful visualization.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.1-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.47-green.svg)

## ✨ Features

- **Real-time Webcam Detection** - Live object detection from your webcam feed
- **5 YOLO Models** - Choose from YOLOv8n/s/m/l/x based on your speed/accuracy needs
- **Adjustable Parameters** - Control confidence threshold, IOU threshold, and image size
- **Live Statistics** - Real-time FPS counter and detection statistics
- **Beautiful UI** - Polished interface with custom styling
- **Colored Bounding Boxes** - Distinct colors for each detected class
- **Performance Optimized** - Cached model loading and efficient processing

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam/Camera
- macOS, Linux, or Windows

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd ~/Desktop/yolo-detection-app
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Verify dependencies are installed:**
   ```bash
   pip list
   ```

### Running the App

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📖 Usage

1. **Select a Model**
   - Choose from YOLOv8n (fastest) to YOLOv8x (most accurate)
   - YOLOv8n is recommended for real-time webcam detection

2. **Adjust Parameters**
   - **Confidence Threshold**: Minimum confidence for detections (default: 0.25)
   - **IOU Threshold**: IoU threshold for NMS (default: 0.45)
   - **Image Size**: Input size for model (320/640/1280)

3. **Set Resolution**
   - Choose webcam resolution (320p/640p/1280p)
   - Lower resolution = faster processing

4. **Start Detection**
   - Click the **▶️ Start** button
   - Grant webcam permissions if prompted
   - View real-time detections with bounding boxes

5. **Monitor Performance**
   - Check FPS counter in the video feed
   - View detection statistics in the sidebar
   - Adjust settings for optimal performance

## 🏗️ Project Structure

```
yolo-detection-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git exclusions
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration constants
│
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLO detection engine
│   ├── webcam.py              # Webcam capture
│   └── utils.py               # Helper functions
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py             # Sidebar controls
│   └── styles.py              # Custom CSS
│
└── models/                     # Auto-downloaded YOLO models
    └── .gitkeep
```

## 📊 Model Comparison

| Model   | Size  | Speed   | mAP   | Use Case                    |
|---------|-------|---------|-------|-----------------------------|
| YOLOv8n | 6 MB  | Fastest | 37.3  | Real-time webcam (10+ FPS)  |
| YOLOv8s | 22 MB | Fast    | 44.9  | Good balance                |
| YOLOv8m | 50 MB | Moderate| 50.2  | Higher accuracy             |
| YOLOv8l | 84 MB | Slow    | 52.9  | Production quality          |
| YOLOv8x | 131 MB| Slowest | 53.9  | Maximum accuracy            |

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Model parameters
- Webcam resolutions
- Color palette
- Performance thresholds

## 🎯 Performance Tips

- **For best FPS**: Use YOLOv8n with 640p resolution
- **For best accuracy**: Use YOLOv8x with 1280p resolution
- **Good balance**: Use YOLOv8s with 640p resolution
- Ensure good lighting for better detection accuracy
- Close other camera applications before starting
- Adjust confidence threshold to reduce false positives

## 🐛 Troubleshooting

**Webcam not working:**
- Check camera permissions in System Preferences (macOS)
- Ensure no other app is using the camera
- Try changing the camera ID in `webcam.py` if you have multiple cameras

**Low FPS:**
- Switch to a smaller model (YOLOv8n)
- Reduce resolution to 320p
- Lower image size to 320

**Model not loading:**
- Check internet connection (models auto-download)
- Verify write permissions in `models/` directory

## 📦 Dependencies

- `ultralytics==8.1.47` - YOLO implementation
- `opencv-python==4.9.0.80` - Computer vision
- `streamlit==1.31.1` - Web framework
- `torch==2.2.1` - Deep learning backend
- `numpy==1.26.4` - Numerical operations
- `pillow==10.2.0` - Image processing

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- Computer vision by [OpenCV](https://opencv.org/)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Enjoy real-time object detection! 🎯**
