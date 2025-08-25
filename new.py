#!/usr/bin/env python3
import os
import time
import csv
import logging
import threading
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from pyfingerprint.pyfingerprint import PyFingerprint

# =========================
# CONFIG
# =========================
MODEL_PATH = '/home/asus/Downloads/IOT_project/best_garbage_model.keras'
CLASS_NAMES = ['biological', 'metal', 'paper', 'plastic']
IP_WEBCAM_URL = 'http://192.168.10.49:8080/shot.jpg'
SCRIPT_URL = 'https://script.google.com/macros/s/REPLACE_WITH_YOUR_WEB_APP_ID/exec'

SERIAL_PORT = '/dev/serial0'
BAUD_RATE   = 57600

LOG_FOLDER = 'logs'
IMAGE_SAVE_FOLDER = os.path.join(LOG_FOLDER, 'images')
CSV_LOG_FILE = os.path.join(LOG_FOLDER, 'local_log.csv')

REQUEST_TIMEOUT_BG = 8.0   # Background thread can wait longer
CYCLE_SLEEP = 0.1
JPEG_QUALITY = 60

# =========================
# SETUP
# =========================
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logging.info("Model loaded.")

# =========================
# FUNCTIONS
# =========================
def prepare_image(img, target_size=(299, 299)):
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def classify_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    x = prepare_image(img)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0])) * 100
    return CLASS_NAMES[idx], round(conf, 2)

def fetch_snapshot():
    r = requests.get(IP_WEBCAM_URL, timeout=2.5)
    r.raise_for_status()
    return r.content

def save_local_log(name, uid, status, img_class, conf, img_bytes=None):
    ts = datetime.now()
    img_file = ts.strftime('%Y-%m-%d_%H-%M-%S.jpg')

    if img_bytes:
        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            img = img.resize((300, 300))
            img.save(os.path.join(IMAGE_SAVE_FOLDER, img_file), format='JPEG', quality=JPEG_QUALITY)
        except Exception as e:
            logging.warning(f"Image save failed: {e}")
            img_file = 'error.jpg'

    new_file = not os.path.exists(CSV_LOG_FILE) or os.path.getsize(CSV_LOG_FILE) == 0
    with open(CSV_LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['Timestamp', 'Day', 'Name', 'UserID', 'Status', 'Class', 'Confidence', 'Image'])
        w.writerow([
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            ts.strftime('%A'),
            name,
            uid,
            status,
            img_class,
            conf,
            img_file if img_bytes else 'N/A'
        ])

def send_to_sheet_async(name, uid, status, img_class, conf):
    """Non-blocking send to Google Sheet in background."""
    data = {
        "name": name,
        "user": str(uid),
        "status": status,
        "img_class": img_class,
        "confidence": str(conf)
    }
    def task():
        try:
            r = requests.post(SCRIPT_URL, data=data, timeout=REQUEST_TIMEOUT_BG)
            r.raise_for_status()
            logging.info(f"[Sheet] Logged {name} | {uid} | {img_class}")
        except Exception as e:
            logging.warning(f"[Sheet] Send failed: {e}")
    threading.Thread(target=task, daemon=True).start()

def init_fingerprint():
    try:
        f = PyFingerprint(SERIAL_PORT, BAUD_RATE, 0xFFFFFFFF, 0x00000000)
        if not f.verifyPassword():
            raise ValueError("Sensor password incorrect")
        logging.info("Fingerprint sensor ready.")
        return f
    except Exception as e:
        logging.error("Fingerprint sensor init failed.")
        raise

def lookup_user(pos):
    if pos < 0:
        return None, None
    return 12100 + pos, f"User{pos}"

def process_cycle(f):
    if f.readImage() is False:
        return False

    t0 = time.perf_counter()
    f.convertImage(0x01)
    pos, acc = f.searchTemplate()

    if pos < 0:
        logging.info("[UNKNOWN] Finger not found")
        save_local_log("Unknown", 0, "Not Active", "N/A", 0.0, None)
        send_to_sheet_async("Unknown", 0, "Not Active", "N/A", 0.0)
        time.sleep(1.0)
        return True

    uid, name = lookup_user(pos)
    logging.info(f"[MATCH] {name} ID={uid} Conf={acc}")

    try:
        img_bytes = fetch_snapshot()
        img_class, conf = classify_image(img_bytes)
    except Exception as e:
        logging.error(f"Capture/classify failed: {e}")
        img_class, conf, img_bytes = "error", 0.0, None

    save_local_log(name, uid, "Active", img_class, conf, img_bytes)
    send_to_sheet_async(name, uid, "Active", img_class, conf)

    logging.info(f"Cycle completed in {time.perf_counter()-t0:.2f}s")
    time.sleep(1.0)
    return True

# =========================
# MAIN
# =========================
def main():
    f = init_fingerprint()
    logging.info("Waiting for finger...")
    while True:
        try:
            if not process_cycle(f):
                time.sleep(CYCLE_SLEEP)
        except KeyboardInterrupt:
            logging.info("Exiting.")
            break
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
