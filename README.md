##🚦 Traffic Violation Detection System

An advanced AI-powered traffic violation detection web app built with Python, YOLO, Streamlit, and EasyOCR. Automatically detect overspeeding, red-light violations, helmet violations, and license plates — all in real-time with a modern UI.

✨ Features

Vehicle Detection: Cars, motorcycles, buses, vans, trucks

Red-Light Detection: Identifies traffic light state (Red / Green)

Stop-Line Violation: Detects vehicles crossing the stop-line on a red light

Speed Detection: Tracks vehicles and estimates their speed using pixel-to-meter calibration

Helmet Detection: Detects riders without helmets (supports optional custom helmet model)

License Plate Detection: Extracts plates and performs OCR using EasyOCR

Vehicle Tracking: Lightweight tracker assigns unique IDs to each vehicle

Process Images & Videos: Upload and analyze in real time

Output Video Export: Download fully processed annotated videos

Custom UX: Gradient UI, cards, neon headings & modern look

🎨 UI Highlights

Elegant gradient background

Beautiful rounded cards with hover animations

Live video frame preview while processing

Neon-style heading

Organized sidebar configuration

Smooth interface with Streamlit components

🚀 Getting Started
Prerequisites

Python 3.8+

pip

GPU optional (YOLO runs on CPU too)

Installation

Clone the repository

git clone https://github.com/yourusername/traffic-violation-detection.git
cd traffic-violation-detection


Install dependencies

pip install -r requirements.txt


or manually:

pip install streamlit ultralytics opencv-python easyocr numpy pillow pandas imageio

Running the App
streamlit run traffic_violation_app.py


The app will open automatically in your browser at:

👉 http://localhost:8501

📁 Project Structure
TrafficViolationSystem/
├── traffic_violation_app.py   # Main application
├── models/                    # Optional YOLO weights
└── README.md                  # Documentation

🛠️ Technologies Used

Python – Core backend

YOLO (Ultralytics) – Object detection

EasyOCR – License plate text extraction

Streamlit – Web-based interface

OpenCV – Image/Video processing

NumPy / Pandas – Data handling

📝 Usage

Upload Image/Video
Upload a file from the sidebar.

Adjust Detection Settings

Confidence

IoU Threshold

Stop-line position

Speed calibration (pixels-per-meter, FPS)

Process and View Output

Bounding boxes

Speed

Violations

Plate numbers

Traffic light status

Video Export
Download the annotated final video.

Data Table Output
Includes detection info for every vehicle:

ID

Vehicle type

Plate

Speed

Violations

🎯 Data Structure

Each detected vehicle is stored as:

{
    "id": <unique_id>,
    "type": "car/motorcycle/bus...",
    "bbox": [x1, y1, x2, y2],
    "plate": "KA05AB1234",
    "speed_kmh": 58.4,
    "violations": "red_light,speed,no_helmet"
}

🤝 Contributing

Contributions are always welcome!
Feel free to fork this project and open pull requests with enhancements.

📄 License

This project is open-source and available under the MIT License.

Made with ❤️ using Streamlit

Developed by Syed Ahamed Ali
