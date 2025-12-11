# 🚦 Traffic Violation Detection System

An advanced AI-powered traffic violation detection web app built with **Python**, **YOLO**, **Streamlit**, and **EasyOCR**. Automatically detect overspeeding, red-light violations, helmet violations, and license plates — all in real-time with a modern UI.

## ✨ Features

- **Vehicle Detection**: Cars, motorcycles, buses, vans, trucks  
- **Red-Light Detection**: Identifies traffic light state (Red / Green)  
- **Stop-Line Violation**: Detects vehicles crossing the stop-line on a red light  
- **Speed Detection**: Tracks vehicles and estimates speed using calibration  
- **Helmet Detection**: Detects riders without helmets  
- **License Plate Detection**: OCR using EasyOCR  
- **Vehicle Tracking**: Unique IDs for each vehicle  
- **Image & Video Processing**: Analyze both formats  
- **Export Processed Video**: Download full annotated output  
- **Modern UI**: Gradient background, neon title, card layout  

## 🎨 UI Highlights

- Soft gradient background  
- Clean modern cards  
- Smooth animations  
- Live video frame preview  
- Sidebar settings  
- Minimal, attractive interface  

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Ultralytics YOLO
- OpenCV

### Installation

```bash
git clone https://github.com/yourusername/traffic-violation-detection.git
cd traffic-violation-detection
pip install -r requirements.txt
Or install manually:

bash
Copy code
pip install streamlit ultralytics opencv-python easyocr numpy pillow pandas imageio
Running the App
bash
Copy code
streamlit run traffic_violation_app.py
The app will run at:

arduino
Copy code
http://localhost:8501
📁 Project Structure
Copy code
TrafficViolationSystem/
├── traffic_violation_app.py
├── models/
└── README.md
🛠️ Technologies Used
Python

YOLO (Ultralytics)

EasyOCR

Streamlit

OpenCV

NumPy / Pandas

📝 Usage
Upload image/video

Adjust confidence, IoU, speed calibration

View detections (speed, violations, plates)

Download processed video

View data table for all vehicles

🎯 Vehicle Data Structure
python
Copy code
{
    "id": <unique_id>,
    "type": "car/motorcycle/bus",
    "bbox": [x1, y1, x2, y2],
    "plate": "TN09AB1234",
    "speed_kmh": 45.8,
    "violations": "red_light,speed,no_helmet"
}
🤝 Contributing
Feel free to fork and submit improvements!

📄 License
Open source under the MIT License

Made with ❤️ using Streamlit

Developed by Syed Ahamed Ali
