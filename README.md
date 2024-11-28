# YOLO Real-Time Object Detection Project

## 📌 Project Overview
This project implements **YOLO (You Only Look Once)**, a state-of-the-art, real-time object detection system. It detects objects in images and videos with high speed and accuracy, making it ideal for real-time applications such as surveillance, self-driving cars, and more.

## 🚀 Features
- Real-time object detection with YOLO.
- Supports webcam input formats.
- Configurable confidence threshold for detections.
- Support YOLO version 11.
- Easy integration with cameras or other video feeds.
- Visualizes detected objects with bounding boxes and labels.

## 🛠️ Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For image and video processing.
- **YOLO Framework**: Core object detection engine.
- **Pre-trained YOLO Models**: Leveraging transfer learning for accurate predictions.

## 📋 Prerequisites
Ensure you have the following installed:
- Python 3.9 or above
- pip (Python package manager)
- OpenCV (`pip install opencv-python`)
- Pre-trained YOLO weights and configuration files.

## 📂 Project Structure
```
yolo-object-detection/
├── README.md              # Project documentation
├── main.py                # Main application script
├── models/                # YOLO model weights and configuration files
│   ├── yolo11l.pt
│   ├── yolo11l-cls.pt
│   ├── yolo11l-seg.pt
│   ├── yolo11l-obb.pt
├── utils.py                 # Utility script
├── images/                 # Input images
├── output/                # Output with detected objects
└── requirements.txt       # Python dependencies
```

## 🔧 Installation

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

### Real-Time Detection (Webcam)
Run the following command to use your webcam:
```bash
python main.py
```
Select option 1 and enter.

### Classify Objects in an Image
Run the following command to process an image:
```bash
python main.py
```
Select option 2 and enter.

### Real-Time Segmentation (Webcam)
Run the following command to process an image:
```bash
python main.py
```
Select option 3 and enter.

## 🖼️ Sample Output
Sample outputs with bounding boxes and labels are saved in the `output/` directory.

## 📈 Performance
- **Speed**: Capable of processing real-time frames at ~30 FPS (depending on hardware and YOLO version). I have used their Large models.
- **Accuracy**: High precision and recall with minimal false positives.

## 🤝 Contributing
Contributions are welcome! If you have suggestions or bug reports, please open an issue or submit a pull request.

## 🔐 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact
For any queries or collaborations, reach out via:
- Email: ahmadnasrullah7833.com


---

🎉 **Happy Coding!** 🎉
