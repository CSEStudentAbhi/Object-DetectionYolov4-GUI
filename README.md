# Object Detection YOLOv4 GUI

A desktop application for real-time object detection using YOLOv4 with a user-friendly graphical interface built with Python and Tkinter.

## 🎯 Features

- **Real-time Object Detection**: Detect objects in images and live camera feed using YOLOv4
- **Multi-Image Support**: Load and process multiple images with navigation controls
- **Live Camera Feed**: Real-time object detection from webcam
- **Interactive Image Viewer**: 
  - Zoom in/out functionality
  - Drag and pan images
  - Image navigation (previous/next)
- **COCO Dataset Support**: Detects 80+ object classes from the COCO dataset
- **User-Friendly Interface**: Clean and intuitive GUI built with Tkinter

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- Tkinter (usually comes with Python)
- PIL (Pillow)
- NumPy

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd Object-DetectionYolov4-GUI
   ```

2. **Install required dependencies**
   ```bash
   pip install opencv-python
   pip install pillow
   pip install numpy
   ```

3. **Download YOLOv4 weights and configuration**
   - Ensure you have the following files in your project directory:
     - `yolov4.weights` (YOLOv4 pre-trained weights)
     - `yolov4.cfg` (YOLOv4 configuration file)
     - `coco.names` (COCO dataset class names)

## 📁 Project Structure

```
Object-DetectionYolov4-GUI/
├── object_detection_gui.py    # Main application file
├── yolov4.weights            # YOLOv4 pre-trained weights
├── yolov4.cfg                # YOLOv4 configuration file
├── coco.names                # COCO dataset class names
├── Data/                     # Sample images directory
│   ├── 10.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   └── car.jpg
└── README.md                 # This file
```

## 🎮 Usage

### Running the Application

```bash
python object_detection_gui.py
```

### Interface Controls

- **Load Images**: Click to select and load multiple images for object detection
- **Start Camera**: Begin real-time object detection from your webcam
- **Stop Camera**: Stop the live camera feed
- **Zoom In/Out**: Adjust the zoom level of displayed images
- **Previous/Next**: Navigate between loaded images
- **Clear**: Clear all loaded images and reset the interface
- **Quit**: Close the application

### Features in Detail

#### Image Processing
- Supports multiple image formats (JPG, PNG, etc.)
- Automatic object detection with bounding boxes
- Confidence threshold of 0.5 for reliable detections
- Non-maximum suppression for clean results

#### Camera Feed
- Real-time object detection from webcam
- Automatic frame processing and display
- Smooth performance with 10ms update intervals

#### Image Navigation
- Load multiple images simultaneously
- Navigate through images with Previous/Next buttons
- Image counter display (e.g., "Image 2 of 5")
- Drag and pan functionality for large images

## 🔧 Technical Details

### Object Detection Algorithm
- **Model**: YOLOv4 (You Only Look Once version 4)
- **Input Size**: 416x416 pixels
- **Confidence Threshold**: 0.5
- **NMS Threshold**: 0.4
- **Dataset**: COCO (Common Objects in Context)

### Performance
- Optimized for real-time processing
- Efficient image scaling and display
- Memory management for multiple images
- Smooth camera feed updates

### Supported Object Classes
The application can detect 80+ object classes including:
- People, animals (cat, dog, horse, etc.)
- Vehicles (car, truck, bus, motorcycle, etc.)
- Common objects (chair, table, book, phone, etc.)
- And many more from the COCO dataset

## 🛠️ Customization

### Adjusting Detection Sensitivity
To modify the detection sensitivity, edit these parameters in `object_detection_gui.py`:

```python
# Confidence threshold (default: 0.5)
if confidence > 0.5:

# NMS threshold (default: 0.4)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

### Changing Input Resolution
To change the input resolution for YOLO:

```python
# Current: 416x416
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure your webcam is connected and not in use by other applications
   - Check camera permissions on your system

2. **Slow performance**
   - Reduce input image resolution
   - Lower the confidence threshold
   - Close other resource-intensive applications

3. **Missing YOLO files**
   - Ensure `yolov4.weights`, `yolov4.cfg`, and `coco.names` are in the project directory
   - Download the files if they're missing

4. **Import errors**
   - Verify all dependencies are installed: `pip install opencv-python pillow numpy`
   - Ensure you're using Python 3.7 or higher

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on the project repository.

---

## Created by
Abhishek Ambi
Sai Koushik YR
Shree Sagar
Siddaraju T

**Note**: This application requires the YOLOv4 weights file (`yolov4.weights`) which is not included in this repository due to size constraints. Please download it separately from the official YOLO repository or other reliable sources.
