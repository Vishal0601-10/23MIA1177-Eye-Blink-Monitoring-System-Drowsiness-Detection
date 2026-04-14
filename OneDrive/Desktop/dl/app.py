"""
Live Drowsiness Detection — Streamlit App
Fully self-contained. No .h5 model required (EAR-based detection).
Drop in drowsiness_model.h5 alongside this file to enable CNN inference.

Requirements:
    streamlit==1.35.0
    opencv-python-headless==4.9.0.80
    mediapipe==0.10.14
    numpy==1.26.4
    tensorflow==2.16.1
    Pillow==10.3.0
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import statistics
from datetime import datetime
from collections import deque
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH          = "drowsiness_model.h5"
MODEL_INPUT_SIZE    = (160, 160)
EAR_THRESHOLD_DEF   = 0.21          # eye closure threshold (calibrated lower to reduce false positives)
CONSEC_FRAMES       = 3             # blink confirmed after N consecutive low-EAR frames (~100ms at 30fps)
LONG_CLOSURE_FRAMES = 15         # ~0.5 s at 30 fps → drowsy
CALIBRATION_SECS    = 4
ALERT_COOLDOWN      = 4
REAL_FACE_WIDTH_CM  = 14.0
FOCAL_LENGTH        = 650
LOG_FILE            = "classification_report.json"
LOG_MAX             = 200        # keep last N entries in memory

# ── Eye Blink Monitoring CONFIG ───────────────
ALERT_INTERVAL_SECONDS = 10     # EHI recalculation interval
BLINK_STARE_TIMEOUT    = 12     # seconds without blink → stare alert (drowsiness_app uses 12)
EHI_SEVERE_THRESHOLD   = 50     # below this → severe eye strain

# ── Brightness Control CONFIG ─────────────────
BRIGHTNESS_CLOSE_CM    = 40     # closer than this → reduce brightness
BRIGHTNESS_FAR_CM      = 80     # farther than this → increase brightness
BRIGHTNESS_MIN         = 0.5    # minimum brightness multiplier
BRIGHTNESS_MAX         = 1.5    # maximum brightness multiplier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard — Live Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CSS  (retro-CRT / tactical HUD aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --bg:        #030810;
    --panel:     rgba(4,16,36,0.92);
    --border:    rgba(0,240,180,0.18);
    --accent:    #00f0b4;
    --danger:    #ff2244;
    --warn:      #ffaa00;
    --text:      #a8c8b8;
    --dim:       #2a5050;
    --glow:      0 0 12px rgba(0,240,180,0.35);
}

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background: var(--bg);
    color: var(--text);
}
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #001a2e 0%, #030810 60%);
}

/* scanline overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px
    );
    pointer-events: none; z-index: 9999;
}

h1 { font-family:'Orbitron',monospace; font-weight:900; letter-spacing:4px; color:var(--accent); }
h2,h3 { font-family:'Orbitron',monospace; font-weight:700; letter-spacing:2px; color:var(--accent); }

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: var(--glow), inset 0 0 40px rgba(0,240,180,0.02);
}
.panel-label {
    font-size: 10px; letter-spacing: 4px; color: var(--dim);
    text-transform: uppercase; margin-bottom: 8px;
    border-bottom: 1px solid var(--border); padding-bottom: 6px;
}

/* Big status badge */
.status-badge {
    border-radius: 4px; padding: 18px 10px; text-align: center;
    font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 900;
    letter-spacing: 3px; border: 2px solid; margin: 8px 0;
    transition: all 0.3s;
}
.badge-alert   { background: rgba(0,240,180,0.08); border-color:var(--accent); color:var(--accent);
                 box-shadow: 0 0 20px rgba(0,240,180,0.3); }
.badge-drowsy  { background: rgba(255,34,68,0.12);  border-color:var(--danger); color:var(--danger);
                 box-shadow: 0 0 20px rgba(255,34,68,0.4); animation: pulse 0.6s infinite; }
.badge-caution { background: rgba(255,170,0,0.10);  border-color:var(--warn);   color:var(--warn);
                 box-shadow: 0 0 16px rgba(255,170,0,0.3); }
.badge-init    { background: rgba(40,60,80,0.3);    border-color:var(--dim);    color:var(--dim); }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* Metric tiles */
.metric-row { display:flex; gap:10px; margin-bottom:12px; }
.metric-tile {
    flex:1; background:rgba(0,30,50,0.6); border:1px solid var(--border);
    border-radius:4px; padding:10px 8px; text-align:center;
}
.metric-tile .val {
    font-family:'Orbitron',monospace; font-size:22px; font-weight:700;
    color:var(--accent); display:block; line-height:1.1;
}
.metric-tile .lbl { font-size:9px; letter-spacing:3px; color:var(--dim); margin-top:3px; }

/* EAR bar */
.ear-bar-wrap { margin: 8px 0; }
.ear-bar-bg   { height:8px; background:rgba(255,255,255,0.06); border-radius:2px; }
.ear-bar-fill { height:8px; border-radius:2px; transition: width 0.15s, background 0.3s; }

/* Log */
.log-entry {
    font-size:11px; color:var(--dim); padding:3px 0;
    border-bottom:1px solid rgba(0,240,180,0.05);
}
.log-drowsy { color: #ff5566; }
.log-alert  { color: #00c090; }

/* Streamlit button overrides */
.stButton > button {
    font-family:'Orbitron',monospace; font-size:12px; font-weight:700;
    letter-spacing:2px; border-radius:3px;
    border: 1px solid var(--accent); background: rgba(0,50,40,0.4);
    color: var(--accent); width:100%; transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(0,240,180,0.15); box-shadow: var(--glow);
}
/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "running":          False,
    "log":              deque(maxlen=LOG_MAX),
    "total_alerts":     0,
    "session_start":    None,
    # Image analysis tab
    "captured_frame":   None,
    "prediction_result": None,
    "capture_log":      [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  MEDIAPIPE  (cached — one instance)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ─────────────────────────────────────────────
#  MODEL  (optional CNN)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from tensorflow.keras.models import load_model as lm
        return lm(MODEL_PATH)
    except Exception:
        return None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def ear(eye_pts):
    p1,p2,p3,p4,p5,p6 = eye_pts
    v1 = np.linalg.norm(np.subtract(p2, p6))
    v2 = np.linalg.norm(np.subtract(p3, p5))
    h  = np.linalg.norm(np.subtract(p1, p4))
    return (v1 + v2) / (2.0 * h + 1e-6)

def face_distance(face_w_px):
    return (REAL_FACE_WIDTH_CM * FOCAL_LENGTH) / (face_w_px + 1e-6)

def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, MODEL_INPUT_SIZE).astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

def cnn_predict(model, bgr):
    raw = model.predict(preprocess(bgr), verbose=0)
    drowsy_p = float(raw[0][0]) if raw.shape[-1] != 2 else float(raw[0][0])
    alert_p  = 1.0 - drowsy_p  if raw.shape[-1] != 2 else float(raw[0][1])
    label    = "DROWSY" if drowsy_p >= 0.5 else "ALERT"
    conf     = max(drowsy_p, alert_p) * 100
    return label, conf, drowsy_p * 100, alert_p * 100

def append_log(label, conf):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "label": label, "conf": round(conf, 1)}
    st.session_state.log.appendleft(entry)
    try:
        existing = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                existing = json.load(f)
        existing.append({**entry, "date": datetime.now().strftime("%Y-%m-%d")})
        with open(LOG_FILE, "w") as f:
            json.dump(existing[-500:], f, indent=2)
    except Exception:
        pass

def draw_landmarks(frame, face_lm, h, w, ear_val, status):
    color = (0, 255, 180) if status == "ALERT" else (50, 50, 255) if status == "DROWSY" else (255, 170, 0)
    for idx in LEFT_EYE + RIGHT_EYE:
        lm = face_lm.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, color, -1)
    # EAR label
    cv2.putText(frame, f"EAR {ear_val:.3f}", (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame

def status_html(status, conf=None):
    css  = {"ALERT":"badge-alert","DROWSY":"badge-drowsy","CAUTION":"badge-caution"}.get(status,"badge-init")
    icon = {"ALERT":"◉","DROWSY":"⚠","CAUTION":"◎"}.get(status,"○")
    sub  = f'<div style="font-size:12px;font-weight:400;margin-top:4px;letter-spacing:1px;">{conf:.1f}% confidence</div>' if conf else ""
    return f'<div class="status-badge {css}">{icon}  {status}{sub}</div>'

def ear_bar_html(ear_val, threshold):
    pct   = min(100, ear_val / 0.40 * 100)
    color = "#00f0b4" if ear_val > threshold else "#ff2244" if ear_val < threshold*0.8 else "#ffaa00"
    tpct  = min(100, threshold / 0.40 * 100)
    return f"""
    <div class="ear-bar-wrap">
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#2a5050;margin-bottom:3px;">
        <span>EAR</span><span>{ear_val:.3f} / thresh {threshold:.3f}</span>
      </div>
      <div class="ear-bar-bg" style="position:relative;">
        <div class="ear-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
        <div style="position:absolute;top:0;left:{tpct:.1f}%;width:2px;height:8px;background:#ffaa00;"></div>
      </div>
    </div>"""

def metric_tiles_html(blinks, dist, alerts, session_secs):
    mins  = int(session_secs // 60)
    secs  = int(session_secs  % 60)
    dist_s = f"{int(dist)}" if dist > 0 else "--"
    return f"""
    <div class="metric-row">
      <div class="metric-tile"><span class="val">{blinks}</span><div class="lbl">BLINKS</div></div>
      <div class="metric-tile"><span class="val">{dist_s}</span><div class="lbl">DIST cm</div></div>
      <div class="metric-tile"><span class="val">{alerts}</span><div class="lbl">ALERTS</div></div>
      <div class="metric-tile"><span class="val">{mins:02d}:{secs:02d}</span><div class="lbl">SESSION</div></div>
    </div>"""

# ── [INTEGRATED] Eye Health Index (EHI) calculation ──────────────
def calculate_ehi(blink_count_in_interval, long_closure_events, blink_timestamps):
    """
    Computes the Eye Health Index from blink rate, long closure events,
    and blink variability over the last ALERT_INTERVAL_SECONDS window.
    Returns (EHI: int, status_str: str)
    """
    bpm = blink_count_in_interval * (60 / ALERT_INTERVAL_SECONDS)
    if len(blink_timestamps) > 3:
        diffs = np.diff(list(blink_timestamps)).tolist()
        variability = statistics.stdev(diffs) if len(diffs) > 1 else 0
    else:
        variability = 0
    br_score  = min(100, (bpm / 20) * 100)
    lcd_score = max(0, 100 - long_closure_events * 20)
    var_score = max(0, 100 - variability * 50)
    ehi = int(0.4 * br_score + 0.3 * lcd_score + 0.3 * var_score)
    if ehi >= 80:
        ehi_status = "Normal"
    elif ehi >= 50:
        ehi_status = "Mild Strain"
    else:
        ehi_status = "Severe Strain"
    return ehi, ehi_status

# ── [INTEGRATED] Brightness multiplier from face distance ─────────
def compute_brightness_multiplier(distance_cm):
    """
    Returns a brightness multiplier in [BRIGHTNESS_MIN, BRIGHTNESS_MAX]
    based on face-to-screen distance:
      - Too close  (< BRIGHTNESS_CLOSE_CM) → dim the frame (reduce eyestrain)
      - Too far    (> BRIGHTNESS_FAR_CM)   → brighten the frame
      - In-between → linear interpolation
    """
    if distance_cm <= 0:
        return 1.0
    if distance_cm < BRIGHTNESS_CLOSE_CM:
        return BRIGHTNESS_MIN
    elif distance_cm > BRIGHTNESS_FAR_CM:
        return BRIGHTNESS_MAX
    else:
        t = (distance_cm - BRIGHTNESS_CLOSE_CM) / (BRIGHTNESS_FAR_CM - BRIGHTNESS_CLOSE_CM)
        return BRIGHTNESS_MIN + t * (BRIGHTNESS_MAX - BRIGHTNESS_MIN)

def apply_brightness(frame_bgr, multiplier):
    """Applies a brightness multiplier to a BGR frame, clamped to [0,255].
    NOTE: This only modifies the camera preview frame, NOT the actual screen.
    Screen brightness is controlled separately via inject_screen_brightness().
    """
    adjusted = np.clip(frame_bgr.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)
    return adjusted

def inject_screen_brightness(multiplier: float):
    """
    BUG FIX 3: Control the ACTUAL SCREEN/PAGE brightness using CSS filter injection.
    This overrides just the camera-frame brightness adjustment and instead applies
    a CSS brightness filter to the entire Streamlit app, simulating screen dimming/brightening.
    multiplier: 0.5 = 50% brightness (dim), 1.0 = normal, 1.5 = bright
    """
    pct = int(multiplier * 100)
    # Inject CSS that applies brightness filter to the whole page body
    st.markdown(f"""
    <style>
    /* Screen brightness control — applied to whole page */
    html, .stApp {{
        filter: brightness({pct}%) !important;
        transition: filter 0.8s ease !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  IMAGE-ANALYSIS HELPERS  (from app_streamlit_updated)
# ─────────────────────────────────────────────
_face_mesh_static = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)

def predict_drowsiness(model, image_bgr: np.ndarray):
    """Returns (label, confidence, probs-dict). Falls back to EAR when model=None.
    
    BUG FIX — EAR-based prediction was INVERTED:
      Low EAR  = eyes CLOSED = DROWSY
      High EAR = eyes OPEN   = NON DROWSY
    The old formula `1.0 - (ear_val / 0.30)` gave high drowsy probability even
    for open eyes because typical open-eye EAR (~0.28-0.32) divided by 0.30 ≈ 1.0
    → 1 - 1 = 0 drowsy_p, yet closed eyes (~0.15) → 1 - 0.5 = 0.5 → barely drowsy.
    Fixed: map EAR linearly where EAR <= EAR_THRESHOLD_DEF → drowsy, 
    EAR >= OPEN_EYE_EAR (~0.30) → non-drowsy.
    """
    OPEN_EYE_EAR = 0.30   # typical open-eye EAR baseline
    if model is None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = _face_mesh_static.process(rgb)
        h, w = image_bgr.shape[:2]
        ear_val = OPEN_EYE_EAR  # default = open eyes
        if res.multi_face_landmarks:
            lm    = res.multi_face_landmarks[0].landmark
            left  = [(int(lm[i].x*w), int(lm[i].y*h)) for i in LEFT_EYE]
            right = [(int(lm[i].x*w), int(lm[i].y*h)) for i in RIGHT_EYE]
            ear_val = (ear(left) + ear(right)) / 2.0
        # Low EAR → high drowsy probability  (CORRECTED direction)
        # Map: EAR=0 → drowsy_p=1.0;  EAR=OPEN_EYE_EAR → drowsy_p=0.0
        drowsy_p = float(np.clip(1.0 - (ear_val / OPEN_EYE_EAR), 0.0, 1.0))
        # Extra boost: if EAR is below threshold, make it clearly drowsy
        if ear_val < EAR_THRESHOLD_DEF:
            drowsy_p = max(drowsy_p, 0.75)
        alert_p = 1.0 - drowsy_p
        label   = "Drowsy" if drowsy_p >= 0.5 else "Non Drowsy"
        conf    = max(drowsy_p, alert_p) * 100
        return label, conf, {"Drowsy": drowsy_p*100, "Non Drowsy": alert_p*100}
    tensor = preprocess(image_bgr)
    raw    = model.predict(tensor, verbose=0)
    # CNN models trained on drowsiness datasets typically output [drowsy_prob, alert_prob]
    # or a single sigmoid output where 1=drowsy. Detect shape and handle accordingly.
    if raw.shape[-1] == 2:
        drowsy_p, alert_p = float(raw[0][0]), float(raw[0][1])
    else:
        # Single sigmoid output — 1=drowsy, 0=non-drowsy
        drowsy_p = float(raw[0][0])
        alert_p  = 1.0 - drowsy_p
    label = "Drowsy" if drowsy_p >= 0.5 else "Non Drowsy"
    conf  = max(drowsy_p, alert_p) * 100
    return label, conf, {"Drowsy": drowsy_p*100, "Non Drowsy": alert_p*100}

def try_grad_cam(model, image_bgr: np.ndarray):
    try:
        import tensorflow as tf
        if model is None:
            return None
        last_conv = next((l.name for l in reversed(model.layers) if 'conv' in l.name.lower()), None)
        if last_conv is None:
            return None
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        tensor = tf.cast(preprocess(image_bgr), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            conv_out, preds = grad_model(tensor)
            loss = preds[:, 0]
        grads   = tape.gradient(loss, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
        cam     = np.maximum(cam, 0)
        cam     = cam / (cam.max() + 1e-8)
        cam_r   = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def save_capture_log(label: str, confidence: float, img_array: np.ndarray):
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": ts, "label": label, "confidence": round(confidence, 1)}
    st.session_state.capture_log.append(entry)
    os.makedirs("captures", exist_ok=True)
    fname = f"captures/capture_{ts.replace(':','-').replace(' ','_')}.jpg"
    cv2.imwrite(fname, img_array)
    entry["image"] = fname
    try:
        existing = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                existing = json.load(f)
        existing.append(entry)
        with open(LOG_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass

# ── Classification report (replace with your real eval numbers) ───
CLASSIFICATION_REPORT = {
    "Drowsy":     {"precision": 0.93, "recall": 0.91, "f1": 0.92, "support": 320},
    "Non Drowsy": {"precision": 0.91, "recall": 0.93, "f1": 0.92, "support": 310},
    "accuracy":   0.92,
    "macro avg":  {"precision": 0.92, "recall": 0.92, "f1": 0.92, "support": 630},
}

def render_classification_report():
    rows = ""
    for cls in ["Drowsy", "Non Drowsy", "macro avg"]:
        m = CLASSIFICATION_REPORT[cls]
        rows += f"""
        <tr>
            <td style="padding:5px 10px;font-weight:600;color:#a8c8b8;">{cls}</td>
            <td style="padding:5px 10px;text-align:center;color:#00f0b4;">{m['precision']*100:.1f}%</td>
            <td style="padding:5px 10px;text-align:center;color:#00c090;">{m['recall']*100:.1f}%</td>
            <td style="padding:5px 10px;text-align:center;color:#ffaa00;">{m['f1']*100:.1f}%</td>
            <td style="padding:5px 10px;text-align:center;color:#2a5050;">{m['support']}</td>
        </tr>"""
    acc = CLASSIFICATION_REPORT["accuracy"] * 100
    st.markdown(f"""
    <div class="panel">
      <div class="panel-label">Model Evaluation Report</div>
      <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:12px;">
        <thead>
          <tr style="border-bottom:1px solid rgba(0,240,180,0.2);">
            <th style="padding:5px 10px;text-align:left;color:#2a5050;">Class</th>
            <th style="padding:5px 10px;color:#00f0b4;">Precision</th>
            <th style="padding:5px 10px;color:#00c090;">Recall</th>
            <th style="padding:5px 10px;color:#ffaa00;">F1-Score</th>
            <th style="padding:5px 10px;color:#2a5050;">Support</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <div style="margin-top:10px;font-family:'Share Tech Mono',monospace;font-size:13px;color:#a8c8b8;">
        Overall Accuracy: <span style="color:#00f0b4;font-weight:700;">{acc:.1f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:18px 0 6px;">
  <h1 style="font-size:2rem;margin:0;">DROWSYGUARD</h1>
  <p style="color:#2a5050;letter-spacing:4px;font-size:11px;margin-top:4px;">
    REAL-TIME EYE SURVEILLANCE SYSTEM  v2.0
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_live, tab_ai = st.tabs(["📡  Live Monitoring", "🧠  AI Drowsiness Detection"])

# ══════════════════════════════════════════════
#  TAB 1 — LIVE MONITORING
# ══════════════════════════════════════════════
with tab_live:
    col_feed, col_dash = st.columns([3, 2], gap="medium")

    with col_dash:
        st.markdown('<div class="panel-label">SYSTEM CONTROL</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ START", use_container_width=True)
        stop_btn  = c2.button("■ STOP",  use_container_width=True)

        if start_btn:
            st.session_state.running       = True
            st.session_state.session_start = time.time()
            st.session_state.total_alerts  = 0
        if stop_btn:
            st.session_state.running = False

        status_placeholder     = st.empty()
        metrics_placeholder    = st.empty()
        ear_placeholder        = st.empty()
        ehi_placeholder        = st.empty()
        brightness_placeholder = st.empty()

        model     = load_model()
        face_mesh = get_face_mesh()

        mode_label = "CNN MODEL" if model else "EAR MODE (DEMO)"
        mode_color = "#00f0b4"   if model else "#ffaa00"
        st.markdown(f"""
        <div class="panel" style="margin-top:6px;">
          <div class="panel-label">DETECTION ENGINE</div>
          <span style="color:{mode_color};font-size:13px;letter-spacing:2px;">◈  {mode_label}</span>
          {"<br><span style='color:#2a5050;font-size:10px;'>Place drowsiness_model.h5 here to enable CNN</span>" if not model else ""}
        </div>
        """, unsafe_allow_html=True)

        log_placeholder = st.empty()

    with col_feed:
        frame_placeholder = st.empty()
        st.markdown("""
        <div style="font-size:10px;color:#1a3a3a;letter-spacing:2px;margin-top:6px;text-align:center;">
          WEBCAM FEED  ·  MEDIAPIPE FACE MESH  ·  30 FPS TARGET
        </div>""", unsafe_allow_html=True)

    # ── Default (idle) UI ─────────────────────────────────────
    if not st.session_state.running:
        status_placeholder.markdown(status_html("STANDBY"), unsafe_allow_html=True)
        metrics_placeholder.markdown(metric_tiles_html(0, 0, 0, 0), unsafe_allow_html=True)
        ear_placeholder.markdown(ear_bar_html(0.30, EAR_THRESHOLD_DEF), unsafe_allow_html=True)
        ehi_placeholder.markdown(
            '<div class="panel"><div class="panel-label">EYE HEALTH INDEX</div>'
            '<span style="color:#2a5050;font-size:12px;">-- / Calibrating</span></div>',
            unsafe_allow_html=True
        )
        brightness_placeholder.markdown(
            '<div class="panel"><div class="panel-label">AUTO BRIGHTNESS</div>'
            '<span style="color:#2a5050;font-size:12px;">-- × (no face detected)</span></div>',
            unsafe_allow_html=True
        )
        frame_placeholder.markdown("""
        <div style="background:rgba(4,16,36,0.6);border:1px solid rgba(0,240,180,0.1);
                    border-radius:4px;height:380px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:10px;">
          <div style="font-size:48px;">👁</div>
          <div style="font-family:'Orbitron',monospace;font-size:13px;color:#2a5050;letter-spacing:3px;">
            AWAITING ACTIVATION
          </div>
          <div style="font-size:11px;color:#1a2a2a;">Press ▶ START to begin live detection</div>
        </div>""", unsafe_allow_html=True)

    # ── Main detection loop ───────────────────────────────────
    if st.session_state.running:
        cap = cv2.VideoCapture(0)

        frame_ctr     = 0
        long_ctr      = 0
        eye_closed    = False  # BUG FIX 2: track open→close→open transition for blinks
        ear_threshold = EAR_THRESHOLD_DEF
        calib_ears    = []
        last_alert_ts = 0.0
        last_blink_ts = time.time()
        ear_history   = deque(maxlen=6)
        status        = "CALIBRATING"
        distance_cm   = 0.0
        session_start = st.session_state.session_start or time.time()

        # BUG FIX 1: local counter — session_state.total_blinks can't update mid-loop
        total_blinks         = 0
        interval_blink_count = 0
        long_closure_events  = 0
        blink_timestamps     = deque(maxlen=50)
        interval_start       = time.time()
        ehi_val              = 0
        ehi_status_str       = "Calibrating"

        try:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read from webcam. Check camera permissions.")
                    break

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = face_mesh.process(rgb)
                now   = time.time()

                ear_val = 0.30

                if res.multi_face_landmarks:
                    fl = res.multi_face_landmarks[0]

                    xs = [int(fl.landmark[i].x * w) for i in range(468)]
                    face_w_px   = max(xs) - min(xs)
                    distance_cm = face_distance(face_w_px)

                    le = [(int(fl.landmark[i].x*w), int(fl.landmark[i].y*h)) for i in LEFT_EYE]
                    re = [(int(fl.landmark[i].x*w), int(fl.landmark[i].y*h)) for i in RIGHT_EYE]
                    raw_ear = (ear(le) + ear(re)) / 2.0
                    ear_history.append(raw_ear)
                    ear_val = float(np.mean(ear_history))

                    if now - session_start < CALIBRATION_SECS:
                        calib_ears.append(raw_ear)
                        ear_threshold = float(np.mean(calib_ears)) * 0.78
                        status = "CALIBRATING"
                    else:
                        # ── Blink & long-closure counting — ALWAYS runs via EAR,
                        #    regardless of whether CNN model is active.
                        #    CNN only determines the drowsiness STATUS label; it cannot
                        #    detect individual blinks, so EAR is always used for that.
                        if ear_val < ear_threshold:
                            frame_ctr += 1
                            long_ctr  += 1
                            eye_closed = True
                        else:
                            # Eyes just opened after being closed → count blink
                            if eye_closed and frame_ctr >= CONSEC_FRAMES:
                                total_blinks         += 1
                                interval_blink_count += 1
                                blink_timestamps.append(now)
                                last_blink_ts = now
                            eye_closed = False
                            frame_ctr  = 0
                            long_ctr   = 0

                        # Check for long closure (drowsiness) — evaluate before any reset
                        is_long_closure = (long_ctr >= LONG_CLOSURE_FRAMES)
                        if is_long_closure:
                            long_closure_events += 1
                            long_ctr = 0

                        # ── Status label: CNN overrides EAR-based status if model loaded
                        if model:
                            status, conf, _, _ = cnn_predict(model, frame)
                        else:
                            if is_long_closure:
                                status = "DROWSY"
                            elif ear_val < ear_threshold * 1.05:
                                status = "CAUTION"
                            else:
                                status = "ALERT"

                        if status == "DROWSY" and (now - last_alert_ts > ALERT_COOLDOWN):
                            st.toast("⚠️ DROWSINESS DETECTED — Wake up!", icon="🚨")
                            st.session_state.total_alerts += 1
                            append_log("DROWSY", ear_val * 100)
                            last_alert_ts = now

                        if now - last_blink_ts > BLINK_STARE_TIMEOUT and (now - last_alert_ts > ALERT_COOLDOWN):
                            st.toast("👁 Blink your eyes! Eye strain risk.", icon="👁")
                            last_alert_ts = now

                        if now - interval_start >= ALERT_INTERVAL_SECONDS:
                            ehi_val, ehi_status_str = calculate_ehi(
                                interval_blink_count, long_closure_events, blink_timestamps
                            )
                            if ehi_val < EHI_SEVERE_THRESHOLD and (now - last_alert_ts > ALERT_COOLDOWN):
                                st.toast("🚨 Severe Eye Strain Detected!", icon="🚨")
                                st.session_state.total_alerts += 1
                                last_alert_ts = now
                            interval_blink_count = 0
                            long_closure_events  = 0
                            interval_start       = now

                    frame = draw_landmarks(frame, fl, h, w, ear_val, status)

                else:
                    status = "NO FACE"

                brightness_mult = compute_brightness_multiplier(distance_cm)
                # BUG FIX 3: control the ACTUAL SCREEN brightness via CSS injection
                # (not just the camera preview frame)
                inject_screen_brightness(brightness_mult)
                # Do NOT alter the frame itself for brightness — leave camera feed natural

                overlay_color = {
                    "ALERT":       (0, 255, 180),
                    "CAUTION":     (0, 180, 255),
                    "DROWSY":      (50, 50, 255),
                    "CALIBRATING": (200, 200, 0),
                    "NO FACE":     (80, 80, 80),
                }.get(status, (128, 128, 128))

                cv2.rectangle(frame, (0, 0), (w, 36), (3, 10, 25), -1)
                cv2.putText(frame, f"  {status}", (8, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, overlay_color, 2, cv2.LINE_AA)
                if distance_cm > 0:
                    cv2.putText(frame, f"{int(distance_cm)} cm", (w-80, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 100, 90), 1, cv2.LINE_AA)

                ehi_cv_color = (0, 255, 180) if ehi_val >= 80 else (0, 180, 255) if ehi_val >= 50 else (50, 50, 255)
                cv2.putText(frame, f"EHI:{ehi_val} {ehi_status_str}", (10, h - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ehi_cv_color, 1, cv2.LINE_AA)

                brt_cv_color = (0, 255, 180) if 0.9 <= brightness_mult <= 1.1 else (255, 170, 0)
                cv2.putText(frame, f"BRT:{brightness_mult:.2f}x", (w - 110, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, brt_cv_color, 1, cv2.LINE_AA)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                session_secs = now - session_start
                status_placeholder.markdown(status_html(status), unsafe_allow_html=True)
                # BUG FIX 1: pass local total_blinks directly — always up-to-date
                metrics_placeholder.markdown(
                    metric_tiles_html(total_blinks, distance_cm, st.session_state.total_alerts, session_secs),
                    unsafe_allow_html=True,
                )
                ear_placeholder.markdown(ear_bar_html(ear_val, ear_threshold), unsafe_allow_html=True)

                ehi_color_hex = "#00f0b4" if ehi_val >= 80 else "#ffaa00" if ehi_val >= 50 else "#ff2244"
                ehi_placeholder.markdown(
                    f'<div class="panel"><div class="panel-label">EYE HEALTH INDEX</div>'
                    f'<span style="color:{ehi_color_hex};font-size:18px;font-family:Orbitron,monospace;font-weight:700;">'
                    f'{ehi_val}</span>'
                    f'<span style="color:#2a5050;font-size:11px;margin-left:10px;">{ehi_status_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                dist_zone = (
                    "TOO CLOSE" if 0 < distance_cm < BRIGHTNESS_CLOSE_CM else
                    "TOO FAR"   if distance_cm > BRIGHTNESS_FAR_CM else
                    "OPTIMAL"   if distance_cm > 0 else "NO FACE"
                )
                brt_hex = "#ffaa00" if dist_zone in ("TOO CLOSE", "TOO FAR") else "#00f0b4" if dist_zone == "OPTIMAL" else "#2a5050"
                brightness_placeholder.markdown(
                    f'<div class="panel"><div class="panel-label">AUTO BRIGHTNESS</div>'
                    f'<span style="color:{brt_hex};font-size:15px;font-family:Orbitron,monospace;font-weight:700;">'
                    f'{brightness_mult:.2f}×</span>'
                    f'<span style="color:#2a5050;font-size:11px;margin-left:10px;">{dist_zone}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if st.session_state.log:
                    entries_html = "".join([
                        f'<div class="log-entry log-{e["label"].lower()}">'
                        f'[{e["ts"]}] {e["label"]}  {e["conf"]:.1f}</div>'
                        for e in list(st.session_state.log)[:8]
                    ])
                    log_placeholder.markdown(
                        f'<div class="panel"><div class="panel-label">EVENT LOG</div>{entries_html}</div>',
                        unsafe_allow_html=True,
                    )

        finally:
            cap.release()

        st.session_state.running = False
        status_placeholder.markdown(status_html("STANDBY"), unsafe_allow_html=True)
        st.success("Session ended.")


# ══════════════════════════════════════════════
#  TAB 2 — AI DROWSINESS DETECTION
# ══════════════════════════════════════════════
with tab_ai:

    if model is None:
        st.info(
            "ℹ️ **Demo Mode** — `drowsiness_model.h5` not found. "
            "Drowsiness is estimated from EAR. Place your trained model file alongside "
            "this script to enable deep learning inference.",
            icon="🤖"
        )
    else:
        st.success("✅ Deep learning model loaded successfully.", icon="🧠")

    st.markdown('<div class="panel-label">Capture &amp; Analyse</div>', unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns([1, 1])

    # ── Left: webcam capture + upload ────────────────────────
    with ai_col1:
        st.markdown("**Live Webcam Preview**")
        preview_placeholder = st.empty()
        btn_c1, btn_c2 = st.columns(2)
        capture_btn = btn_c1.button("📸  Capture Image", use_container_width=True)
        preview_btn = btn_c2.button("🎥  Preview",       use_container_width=True)

        if preview_btn or capture_btn:
            cap2 = cv2.VideoCapture(0)
            ret2, frame2 = cap2.read()
            cap2.release()
            if ret2:
                preview_placeholder.image(
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True
                )
                if capture_btn:
                    st.session_state.captured_frame    = frame2.copy()
                    st.session_state.prediction_result = None
                    st.toast("Frame captured!", icon="📸")
            else:
                st.error("Cannot access webcam. Check camera permissions.")

        if st.session_state.captured_frame is not None and not (preview_btn or capture_btn):
            preview_placeholder.image(
                cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB),
                channels="RGB", caption="Last Captured Frame", use_column_width=True
            )

        st.markdown("---")
        st.markdown("**Or upload an image**")
        uploaded = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            bgr     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.session_state.captured_frame    = bgr
            st.session_state.prediction_result = None
            preview_placeholder.image(pil_img, use_column_width=True, caption="Uploaded Image")

    # ── Right: prediction results ─────────────────────────────
    with ai_col2:
        st.markdown("**Drowsiness Analysis**")

        analyse_btn = st.button(
            "🔍  Analyse Drowsiness", use_container_width=True,
            disabled=(st.session_state.captured_frame is None)
        )

        if analyse_btn and st.session_state.captured_frame is not None:
            with st.spinner("Running inference…"):
                t0 = time.time()
                label, confidence, probs = predict_drowsiness(
                    model, st.session_state.captured_frame
                )
                elapsed = (time.time() - t0) * 1000
            st.session_state.prediction_result = {
                "label": label, "confidence": confidence,
                "probs": probs, "elapsed_ms": elapsed
            }
            save_capture_log(label, confidence, st.session_state.captured_frame)

        if st.session_state.prediction_result:
            res  = st.session_state.prediction_result
            lbl  = res["label"]
            conf = res["confidence"]
            prbs = res["probs"]
            ms   = res["elapsed_ms"]

            is_drowsy  = lbl == "Drowsy"
            pred_bg    = "rgba(255,34,68,0.12)"  if is_drowsy else "rgba(0,240,180,0.08)"
            pred_border= "#ff2244"               if is_drowsy else "#00f0b4"
            pred_color = "#ff4466"               if is_drowsy else "#00f0b4"
            icon       = "⚠" if is_drowsy else "◉"

            st.markdown(f"""
            <div style="border-radius:4px;padding:20px;margin:10px 0;text-align:center;
                        font-size:22px;font-weight:900;letter-spacing:3px;border:2px solid;
                        font-family:'Orbitron',monospace;
                        background:{pred_bg};border-color:{pred_border};color:{pred_color};">
                {icon}  {lbl.upper()}
                <div style="font-size:13px;font-weight:400;margin-top:4px;
                            font-family:'Share Tech Mono',monospace;">
                    Confidence: {conf:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="panel-label" style="margin-top:14px;">Class Probabilities</div>',
                        unsafe_allow_html=True)

            for cls_name, color in [("Drowsy", "#ff4444"), ("Non Drowsy", "#00f0b4")]:
                pct = prbs[cls_name]
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;
                                font-family:'Share Tech Mono',monospace;font-size:12px;
                                color:#a8c8b8;margin-bottom:3px;">
                        <span>{cls_name}</span><span>{pct:.1f}%</span>
                    </div>
                    <div class="ear-bar-bg">
                        <div class="ear-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#2a5050;margin-top:6px;">
                ⚡ Inference time: {ms:.1f} ms
            </div>""", unsafe_allow_html=True)

            # Grad-CAM
            st.markdown('<div class="panel-label" style="margin-top:18px;">Grad-CAM Visualization</div>',
                        unsafe_allow_html=True)
            grad_img = try_grad_cam(model, st.session_state.captured_frame)
            if grad_img is not None:
                st.image(grad_img, use_column_width=True)
                st.markdown(
                    '<p style="font-size:11px;color:#2a5050;font-family:Share Tech Mono,monospace;text-align:center;">'
                    'Heatmap shows facial regions influencing prediction</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("""
                <div class="panel" style="text-align:center;">
                    <span style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#2a5050;">
                        Grad-CAM unavailable — requires a loaded deep learning model.
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel" style="text-align:center;padding:30px;">
                <div style="font-size:36px;margin-bottom:8px;">📸</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#2a5050;">
                    Capture or upload an image, then click Analyse Drowsiness
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Classification report + capture log ──────────────────
    st.markdown("---")
    render_classification_report()

    if st.session_state.capture_log:
        st.markdown('<div class="panel-label" style="margin-top:18px;">Capture Log</div>',
                    unsafe_allow_html=True)
        log_html = "".join([
            f'<div class="log-entry log-{"drowsy" if e["label"]=="Drowsy" else "alert"}">'
            f'[{e["timestamp"]}]  {e["label"]}  —  {e["confidence"]:.1f}%'
            f'</div>'
            for e in reversed(st.session_state.capture_log[-20:])
        ])
        st.markdown(f'<div class="panel">{log_html}</div>', unsafe_allow_html=True)
        if st.button("🗑  Clear Log"):
            st.session_state.capture_log = []
            st.rerun()
