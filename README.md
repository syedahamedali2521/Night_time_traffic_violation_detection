🚦 Traffic Violation Detection System
YOLO + Streamlit + EasyOCR | Helmet, Speed, Red-Light, & License Plate Detection

This project is an advanced AI-powered Traffic Violation Detection System built using YOLO, Streamlit, EasyOCR, and OpenCV.
It automatically detects speeding, red-light jumping, helmet violations, license plates, and tracks vehicles across frames.

🧠 Features
✅ Vehicle Detection

Detects multiple vehicle types such as:

Cars

Trucks

Buses

Motorcycles

Vans

🟥 Red-Light Violation Detection

Detects traffic light state (Red/Green)

Determines if vehicle crosses the stop-line during a red light

🏍️ Helmet Violation Detection

Detects motorcycle riders

Identifies presence/absence of helmets

Supports optional separate helmet YOLO model

🚗 License Plate Detection & OCR

Detects vehicle license plates

Extracts characters using EasyOCR

Plate enhancement preprocessing included

📈 Speed Estimation

Uses object tracking across frames

Converts pixel displacement to m/s → km/h

Customizable pixels-per-meter calibration

Speed-limit violation detection

🎯 Object Tracking

Lightweight IoU + centroid tracker

Assigns unique IDs to vehicles

Maintains tracks for realistic speed calculation

🎥 Image + Video Support

Upload images

Upload & process videos

Export processed video

Real-time preview while processing

🗂️ Project Structure
traffic_violation_app.py   # Main Streamlit App
README.md                  # Project Documentation
models/                    # (Optional) YOLO weights

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/traffic-violation-detection.git
cd traffic-violation-detection

2️⃣ Create virtual environment (optional)
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows

3️⃣ Install dependencies
pip install -r requirements.txt


If you don't have a requirements file yet, install manually:

pip install streamlit opencv-python ultralytics easyocr pillow numpy pandas imageio

▶️ Running the Application

Launch the Streamlit app:

streamlit run traffic_violation_app.py


You will see the web interface in your browser (usually at):
👉 http://localhost:8501/

🎛️ App Parameters
Parameter	Description
Confidence	YOLO detection confidence
IoU	Non-max suppression IoU threshold
Stop-Line Y	Vertical pixel position of the traffic stop-line
FPS	Video frames per second (for speed calculation)
Pixels per meter	Calibration value for speed estimation
Speed limit (km/h)	Flag speeding vehicles
🧪 Supported File Types
📸 Images

.jpg, .jpeg, .png

🎞️ Videos

.mp4, .mkv, .avi, .webm, .mov

📊 Output

For each vehicle, the system returns:

Vehicle ID

Vehicle Type

Bounding Box

License Plate Number

Speed (km/h)

Violations detected (comma separated)

Example:

id	type	plate	speed_kmh	violations
3	car	KA03NB1234	58.2
