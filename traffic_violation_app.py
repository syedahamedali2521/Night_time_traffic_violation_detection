# traffic_violation_app.py
# Updated Streamlit app with helmet, speed, red-light, plate OCR and tracking
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import tempfile
import time
import io
import os
import imageio
import base64
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Traffic Violation Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_TITLE = "Traffic Violation Detection System 🚦"

DEFAULT_STOP_LINE_Y = 400
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
IMG_SIZE = 1280

IMAGE_TYPES = ["png", "jpg", "jpeg"]
VIDEO_TYPES = ["mp4", "mov", "avi", "webm", "mkv"]

# ---------------------------
# Inject Gradient CSS (kept from your original)
# ---------------------------
def inject_css():
    css = """
    <style>
    .stApp {
        background: linear-gradient(120deg, #ff3b3b, #22dd33);
        background-attachment: fixed;
    }
    .card {
        background: rgba(255,255,255,0.07);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .card:hover {
        transform: translateY(-4px);
        transition: 0.2s;
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
    }
    .neon {
        color: white;
        text-shadow: 0 0 18px rgba(0,255,0,0.5);
        text-align: center;
        font-size: 32px;
        font-weight: 700;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ---------------------------
# Lightweight Tracker
# ---------------------------
class SimpleTracker:
    """Simple centroid + IoU tracker that assigns incremental IDs."""
    def __init__(self, max_lost=12, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = {}  # id -> bbox
        self.lost = {}     # id -> lost_count
        self.tracks = {}   # id -> deque of centroids (for speed)
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        union = boxAArea + boxBArea - interArea
        if union == 0:
            return 0.0
        return interArea / union

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        detections: list of dict with 'bbox' key [x1,y1,x2,y2] and other meta
        returns mapping id -> detection dict (augmented)
        """
        assigned = {}
        det_bboxes = [d["bbox"] for d in detections]

        # If no existing objects, register all
        if len(self.objects) == 0:
            for det in detections:
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = det["bbox"]
                self.lost[oid] = 0
                cent = ((det["bbox"][0]+det["bbox"][2])//2, (det["bbox"][1]+det["bbox"][3])//2)
                self.tracks[oid] = deque([cent], maxlen=30)
                assigned[oid] = det
            return assigned

        # Compute IoU matrix
        object_ids = list(self.objects.keys())
        iou_matrix = np.zeros((len(object_ids), len(det_bboxes)), dtype=np.float32)
        for i, oid in enumerate(object_ids):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = self.iou(self.objects[oid], db)

        # Greedy matching by IoU
        used_rows, used_cols = set(), set()
        for _ in range(min(iou_matrix.shape)):
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i, j = idx
            if iou_matrix[i, j] < self.iou_threshold:
                break
            if i in used_rows or j in used_cols:
                iou_matrix[i, j] = -1
                continue
            used_rows.add(i); used_cols.add(j)
            oid = object_ids[i]
            det = detections[j]
            # update object
            self.objects[oid] = det["bbox"]
            self.lost[oid] = 0
            cent = ((det["bbox"][0]+det["bbox"][2])//2, (det["bbox"][1]+det["bbox"][3])//2)
            if oid not in self.tracks:
                self.tracks[oid] = deque(maxlen=30)
            self.tracks[oid].append(cent)
            assigned[oid] = det
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # Unmatched detections -> register new objects
        for j, det in enumerate(detections):
            if j in used_cols:
                continue
            oid = self.next_object_id
            self.next_object_id += 1
            self.objects[oid] = det["bbox"]
            self.lost[oid] = 0
            cent = ((det["bbox"][0]+det["bbox"][2])//2, (det["bbox"][1]+det["bbox"][3])//2)
            self.tracks[oid] = deque([cent], maxlen=30)
            assigned[oid] = det

        # Increase lost count for unmatched existing objects
        for i, oid in enumerate(object_ids):
            if i in used_rows:
                continue
            self.lost[oid] += 1

        # Deregister lost objects
        remove_ids = [oid for oid, cnt in self.lost.items() if cnt > self.max_lost]
        for oid in remove_ids:
            del self.objects[oid]
            del self.lost[oid]
            if oid in self.tracks:
                del self.tracks[oid]

        return assigned

    def get_track(self, oid):
        return list(self.tracks.get(oid, []))


# ---------------------------
# Load YOLO + EasyOCR
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_models(weights=None, helmet_weights=None):
    # Load main detector
    if weights and os.path.exists(weights):
        model = YOLO(weights)
    else:
        model = YOLO("yolov8n.pt")  # default small model

    # Optionally load helmet-only model if provided (separate weights)
    helmet_model = None
    if helmet_weights and os.path.exists(helmet_weights):
        helmet_model = YOLO(helmet_weights)

    ocr_model = easyocr.Reader(['en'], gpu=False)
    return model, helmet_model, ocr_model

# ---------------------------
# OCR helpers
# ---------------------------
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr

def read_plate(ocr, crop):
    try:
        thr = preprocess_plate(crop)
        text = ocr.readtext(thr, detail=0)
        if not text:
            text = ocr.readtext(crop, detail=0)
        text = "".join(text).upper().replace(" ", "")
        return "".join([c for c in text if c.isalnum()])
    except Exception:
        return ""

# ---------------------------
# Traffic Light State (improved)
# ---------------------------
def detect_light(frame, bboxes):
    if not bboxes:
        return "unknown"
    # choose largest traffic light bbox
    bboxes_sorted = sorted(bboxes, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
    (x1,y1,x2,y2), _ = bboxes_sorted[0]
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if y2<=y1 or x2<=x1:
        return "unknown"
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0,120,120), (10,255,255))
    red2 = cv2.inRange(hsv, (170,120,120), (180,255,255))
    green = cv2.inRange(hsv, (40,40,40), (90,255,255))

    r = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    g = cv2.countNonZero(green)
    if r > g and r > 50: return "red"
    if g > r and g > 50: return "green"
    return "unknown"

# ---------------------------
# Utility
# ---------------------------
def box_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)//2, (y1+y2)//2)

def box_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

# ---------------------------
# Core Detection & Violation Logic
# ---------------------------
def process_frame(frame, model: YOLO, helmet_model, ocr, tracker: SimpleTracker,
                  stop_line_y, conf_thres, iou_thres, pixels_per_meter, fps, speed_limit_kmh):
    results = model.predict(frame, imgsz=IMG_SIZE, conf=conf_thres, iou=iou_thres, verbose=False)[0]

    vehicles = []
    plates = []
    lights = []
    helmets = []  # from main model if available

    # parse detections
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1,y1,x2,y2 = map(int, box.tolist())
        lbl = model.model.names[int(cls)].lower()

        if "traffic light" in lbl:
            lights.append(([x1,y1,x2,y2], float(conf)))
        elif any(v in lbl for v in ["car","bus","truck","motorcycle","van"]):
            vehicles.append({"bbox":[x1,y1,x2,y2], "type":lbl, "confidence":float(conf), "plate":None, "violation_list":[], "speed_kmh":None})
        elif "plate" in lbl or "license" in lbl:
            plates.append(([x1,y1,x2,y2], float(conf)))
        elif "helmet" in lbl:
            helmets.append(([x1,y1,x2,y2], float(conf)))

    # traffic light state
    tl_state = detect_light(frame, lights)

    # Associate plates to vehicles (if plate detector present)
    for v in vehicles:
        x1,y1,x2,y2 = v["bbox"]
        for (px1,py1,px2,py2), score in plates:
            # plate contained in vehicle bbox (loosen condition slightly)
            if px1 >= x1-10 and py1 >= y1-10 and px2 <= x2+10 and py2 <= y2+10:
                crop = frame[max(0,py1):py2, max(0,px1):px2]
                if crop.size == 0: 
                    continue
                v["plate"] = read_plate(ocr, crop)

    # If helmet_model is provided, run it on motorcycle ROIs to detect helmets
    helmet_from_model = []
    if helmet_model:
        # prepare crops of motorcycle regions for helmet model to detect within full frame
        # we will run helmet_model once on the whole frame to get helmet detections
        hres = helmet_model.predict(frame, imgsz=IMG_SIZE, conf=conf_thres, iou=iou_thres, verbose=False)[0]
        for box, conf, cls in zip(hres.boxes.xyxy, hres.boxes.conf, hres.boxes.cls):
            x1,y1,x2,y2 = map(int, box.tolist())
            lbl = helmet_model.model.names[int(cls)].lower()
            if "helmet" in lbl:
                helmet_from_model.append(([x1,y1,x2,y2], float(conf)))

    # Combine helmet detections: prefer main model's helmet class, then helmet_model
    all_helmets = helmets + helmet_from_model

    # Violation checks
    # 1) Red-light / stop-line crossing: check center Y relative to stop_line_y. 
    #    Orientation: stop_line_y is the Y coordinate of the line on image (0 top). 
    #    We consider crossing if center_y < stop_line_y for approaching from bottom->up or > depending on camera.
    #    We'll treat crossing as center_y < stop_line_y if camera is pointing forward and vehicles move down->up.
    #    We'll allow user to configure "cross direction" if necessary.
    # 2) Speed: compute using tracker displacements and pixels_per_meter & fps
    # 3) No-helmet: for motorcycles only, check overlap with helmet detections or absence thereof
    dets_for_tracker = []
    for v in vehicles:
        dets_for_tracker.append({"bbox": v["bbox"], "meta": v})

    assigned = tracker.update(dets_for_tracker)

    total_violations = 0
    # evaluate each tracked assignment
    for oid, det in assigned.items():
        v = det["meta"]
        bbox = det["bbox"]
        x1,y1,x2,y2 = bbox
        cx, cy = box_center(bbox)
        # compute speed if we have track history
        path = tracker.get_track(oid)  # list of centroids
        speed_kmh = None
        if len(path) >= 2 and pixels_per_meter > 0 and fps > 0:
            # use first and last points in the track for displacement
            (x0,y0) = path[0]
            (xN,yN) = path[-1]
            dx = xN - x0
            dy = yN - y0
            pixel_dist = np.sqrt(dx*dx + dy*dy)
            meters = pixel_dist / pixels_per_meter
            seconds = len(path) / fps  # approximate time window
            if seconds > 0:
                mps = meters / seconds
                speed_kmh = mps * 3.6
                v["speed_kmh"] = speed_kmh

        # Speed violation
        if speed_kmh is not None and speed_kmh > speed_limit_kmh:
            v["violation_list"].append("speed")
            total_violations += 1

        # Red-light / stop-line crossing violation:
        # determine if vehicle has crossed the stop line while light is red.
        # For this, check if previously it was before the line and now beyond it.
        # We'll approximate with centroid y relative to stop_line_y.
        # If approach direction is top->bottom (most common), crossing when cy > stop_line_y.
        # We'll assume crossing when cy > stop_line_y and tl_state == 'red'
        if tl_state == "red":
            if cy > stop_line_y:
                v["violation_list"].append("red_light")
                total_violations += 1

        # No-helmet for motorcycles
        if "motorcycle" in v["type"] or "motorbike" in v["type"] or "bike" in v["type"]:
            # check if any helmet bbox overlaps the rider head area
            head_region = [int(x1 + (x2-x1)*0.2), int(y1), int(x2 - (x2-x1)*0.2), int(y1 + (y2-y1)*0.45)]
            has_helmet = False
            for (hx1,hy1,hx2,hy2), score in all_helmets:
                # check IoU between helmet bbox and head_region
                iou_val = SimpleTracker.iou(head_region, [hx1,hy1,hx2,hx2+0]) if False else 0
                # use simple overlap check
                inter_x1 = max(head_region[0], hx1)
                inter_y1 = max(head_region[1], hy1)
                inter_x2 = min(head_region[2], hx2)
                inter_y2 = min(head_region[3], hy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    has_helmet = True
                    break
            if not has_helmet:
                v["violation_list"].append("no_helmet")
                total_violations += 1

        # Add summary label if plate missing
        if v.get("plate") is None or v.get("plate") == "":
            v["violation_list"].append("unreadable_plate")

    # Draw results
    out = frame.copy()
    for oid, det in assigned.items():
        v = det["meta"]
        bbox = det["bbox"]
        x1,y1,x2,y2 = bbox
        center = box_center(bbox)
        color = (0,255,0)
        if v["violation_list"]:
            color = (0,0,255)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"ID:{oid} {v['type']}"
        if v.get("speed_kmh") is not None:
            label += f" {int(v['speed_kmh'])}km/h"
        cv2.putText(out, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw plate if exists
        if v.get("plate"):
            cv2.putText(out, v["plate"], (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # draw violations as badges
        if v["violation_list"]:
            txt = ",".join(v["violation_list"])
            cv2.putText(out, txt, (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # draw stop line
    cv2.line(out, (0,stop_line_y), (frame.shape[1], stop_line_y), (255,0,0), 3)
    # draw traffic light state
    cv2.putText(out, f"TL: {tl_state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    # prepare vehicles list for DataFrame output
    vehicles_out = []
    for oid, det in assigned.items():
        v = det["meta"]
        vehicles_out.append({
            "id": oid,
            "type": v["type"],
            "bbox": v["bbox"],
            "plate": v.get("plate"),
            "speed_kmh": None if v.get("speed_kmh") is None else round(v.get("speed_kmh"),1),
            "violations": ",".join(v["violation_list"])
        })

    return out, vehicles_out, total_violations, tl_state

# ============================================================
# UI — Streamlit Layout
# ============================================================
st.markdown(f"<h1 class='neon'>{APP_TITLE}</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image or Video", type=IMAGE_TYPES+VIDEO_TYPES)
    weights_path = st.text_input("YOLO Weights (optional)", "")
    helmet_weights_path = st.text_input("Helmet Model Weights (optional)", "")
    conf = st.slider("Confidence", 0.1, 1.0, DEFAULT_CONFIDENCE)
    iou = st.slider("IoU", 0.1, 1.0, DEFAULT_IOU)
    stop_line_y = st.slider("Stop-Line Y Position", 50, 2000, DEFAULT_STOP_LINE_Y)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### Speed / Calibration")
    fps = st.number_input("Video FPS (for speed calc)", min_value=1.0, max_value=120.0, value=25.0, step=1.0)
    pixels_per_meter = st.number_input("Pixels per meter (calibration)", min_value=0.1, max_value=10000.0, value=50.0, step=1.0,
                                       help="Enter how many pixels correspond to 1 meter (approx).")
    speed_limit_kmh = st.number_input("Speed Limit (km/h)", min_value=10, max_value=300, value=50, step=5)
    st.write("Tips:")
    st.write("- Provide calibration for meaningful speed estimates.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Load models once
# ---------------------------
model, helmet_model, ocr = load_models(weights_path.strip() if weights_path.strip() else None,
                                      helmet_weights_path.strip() if helmet_weights_path.strip() else None)

# create tracker instance
tracker = SimpleTracker(max_lost=15, iou_threshold=0.2)

# ============================================================
# AUTO-RUN LOGIC
# ============================================================
if uploaded_file:

    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_video = file_ext in VIDEO_TYPES

    # ----------------------------
    # IMAGE MODE
    # ----------------------------
    if not is_video:
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        out, vehicles, violations, tl = process_frame(
            img, model, helmet_model, ocr, tracker,
            stop_line_y, conf, iou, pixels_per_meter, fps, speed_limit_kmh
        )

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
        st.write(f"### Violations detected (approx): {violations}")
        if vehicles:
            st.write(pd.DataFrame(vehicles))

    # ----------------------------
    # VIDEO MODE (AUTO-RUN)
    # ----------------------------
    else:
        st.info("⏳ Processing video… please wait. This may take a while depending on model size.")
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix="."+file_ext)
        temp_in.write(uploaded_file.read())
        temp_in.flush()

        cap = cv2.VideoCapture(temp_in.name)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        writer = cv2.VideoWriter(temp_out.name, fourcc, video_fps, (video_w, video_h))

        frame_skip = st.slider("Process every Nth frame", 1, 10, 2)

        progress = st.progress(0)
        live = st.empty()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx = 0
        processed = 0
        total_violations = 0
        vehicles_log = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            progress.progress(min(idx/frame_count, 1.0))

            if idx % frame_skip != 0:
                # still write unprocessed frames to keep timing consistent, or skip
                writer.write(frame)
                continue

            out, vehicles, violations, tl = process_frame(
                frame, model, helmet_model, ocr, tracker,
                stop_line_y, conf, iou, pixels_per_meter, video_fps if video_fps>0 else fps, speed_limit_kmh
            )

            # live preview
            live.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

            # add to output video
            writer.write(out)
            processed += 1
            total_violations += violations
            vehicles_log.extend(vehicles)

        cap.release()
        writer.release()

        st.success("Video processed successfully!")
        st.write(f"### Total violations detected (approx): {total_violations}")
        if vehicles_log:
            df = pd.DataFrame(vehicles_log)
            st.write(df.drop_duplicates(subset=["id"]).reset_index(drop=True))

        st.video(temp_out.name)
        with open(temp_out.name, "rb") as f:
            st.download_button(
                "Download Processed Video",
                data=f,
                file_name="processed_output.mp4",
                mime="video/mp4"
            )
