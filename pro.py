import os
import time
import csv
import logging
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
IP_WEBCAM_URL = 'http://192.168.10.49:8080/shot.jpg'   # change to your phone's IP Camera stream snapshot URL
SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzEPgNeJ4UDnaLqGd4vhGg4ze10GoNHOKIcejWae4YzPGdRFrHJsD5vb24GKRs2Gg6_/exec'  # <-- replace

SERIAL_PORT = '/dev/serial0'   # or '/dev/ttyS0' depending on your Pi overlay
BAUD_RATE   = 57600

LOG_FOLDER = 'logs'
IMAGE_SAVE_FOLDER = os.path.join(LOG_FOLDER, 'images')
CSV_LOG_FILE = os.path.join(LOG_FOLDER, 'local_log.csv')

REQUEST_TIMEOUT = 2.5    # seconds (IP cam + Apps Script)
PRED_TARGET_SIZE = (299, 299)
JPEG_SAVE_QUALITY = 60
CYCLE_SLEEP = 0.1         # polling delay when no finger

# Create needed folders
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)

# =========================
# SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Loading TensorFlow model...")
# compile=False for faster load, we only need inference
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logging.info("Model loaded.")

def prepare_image(img: Image.Image, target_size=PRED_TARGET_SIZE):
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def classify_image_from_bytes(image_bytes: bytes):
    """Return (class_name, confidence_pct, pil_image)"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    x = prepare_image(img)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0])) * 100.0
    return CLASS_NAMES[idx], round(conf, 2), img

def fetch_ipcam_snapshot():
    """Fetch snapshot from IP Camera with tight timeout."""
    r = requests.get(IP_WEBCAM_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def send_to_sheet(name: str, user_id: int, status: str, img_class: str, confidence: float):
    """POST to Apps Script. Keep payload tiny & fast."""
    data = {
        'name': name,
        'user': str(user_id),
        'status': status,           # e.g., 'Active'
        'img_class': img_class,     # e.g., 'plastic'
        'confidence': str(confidence) # string for Apps Script convenience
    }
    r = requests.post(SCRIPT_URL, data=data, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def save_local_log(name: str, user_id: int, status: str, img_class: str, confidence: float, image_bytes: bytes|None):
    ts = datetime.now()
    day = ts.strftime('%A')
    img_filename = ts.strftime('%Y-%m-%d_%H-%M-%S.jpg')

    # Save compressed image (optional)
    if image_bytes:
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img = img.resize((300, 300))
            img.save(os.path.join(IMAGE_SAVE_FOLDER, img_filename), format='JPEG', quality=JPEG_SAVE_QUALITY)
        except Exception as e:
            logging.warning(f"Image save failed: {e}")
            img_filename = 'error.jpg'

    # CSV append
    new_file = not os.path.exists(CSV_LOG_FILE) or os.path.getsize(CSV_LOG_FILE) == 0
    with open(CSV_LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['Timestamp', 'Day', 'Name', 'UserID', 'Status', 'Class', 'Confidence', 'ImageFile'])
        w.writerow([
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            day,
            name,
            user_id,
            status,
            img_class,
            confidence,
            img_filename if image_bytes else 'N/A'
        ])

def init_fingerprint_sensor():
    """Init fingerprint sensor and verify password."""
    try:
        fp = PyFingerprint(SERIAL_PORT, BAUD_RATE, 0xFFFFFFFF, 0x00000000)
        if not fp.verifyPassword():
            raise ValueError("Fingerprint sensor password is wrong")
        logging.info("Fingerprint sensor ready.")
        # Optionally reduce the internal security level for speed (depends on module)
        # fp.setSystemParameter(fp.SYSTEM_PARAMETER_SECURITY_LEVEL, 1)
        return fp
    except Exception as e:
        logging.error("Could not initialize fingerprint sensor. Check wiring and serial port.")
        logging.exception(e)
        raise

def lookup_user_from_position(template_position: int):
    """Map template position to a user ID + name. Adjust to your scheme."""
    if template_position < 0:
        return None, None
    user_id = 12100 + template_position
    name = f"User{template_position}"
    return user_id, name

def process_one_cycle(fp: PyFingerprint):
    """
    Wait for a finger; if matched:
      - capture IP cam image
      - classify
      - log to Google Sheet
    Designed to finish in <= ~6s after finger touch.
    Returns True if a full cycle ran, else False.
    """
    # Wait for finger presence
    if fp.readImage() is False:
        # no finger yet
        return False

    t0 = time.perf_counter()
    # Convert & search template
    fp.convertImage(0x01)
    result = fp.searchTemplate()
    position, accuracy = result[0], result[1]

    if position < 0:
        logging.info("[UNKNOWN] Finger not found in DB")
        # log unknown without image
        save_local_log("Unknown", 0, "Not Active", "N/A", 0.0, None)
        try:
            send_to_sheet("Unknown", 0, "Not Active", "N/A", 0.0)
        except Exception as e:
            logging.warning(f"Apps Script (unknown) send failed: {e}")
        # small delay to avoid instant re-trigger
        time.sleep(1.5)
        return True

    user_id, name = lookup_user_from_position(position)
    logging.info(f"[MATCH] {name} | ID: {user_id} | Conf: {accuracy}")

    # Fetch snapshot quickly
    try:
        image_bytes = fetch_ipcam_snapshot()
    except Exception as e:
        logging.error(f"Snapshot fetch failed: {e}")
        # Still log the user activity as Active (no image classification)
        save_local_log(name, user_id, "Active", "snapshot_error", 0.0, None)
        try:
            send_to_sheet(name, user_id, "Active", "snapshot_error", 0.0)
        except Exception as ee:
            logging.warning(f"Apps Script send failed (no snapshot): {ee}")
        time.sleep(1.0)
        return True

    # Classify
    try:
        img_class, confidence, pil_img = classify_image_from_bytes(image_bytes)
    except Exception as e:
        logging.error(f"Classification failed: {e}")
        img_class, confidence = "classify_error", 0.0

    # Log locally + to sheet
    save_local_log(name, user_id, "Active", img_class, confidence, image_bytes)
    try:
        send_to_sheet(name, user_id, "Active", img_class, confidence)
    except Exception as e:
        logging.warning(f"Apps Script send failed: {e}")

    elapsed = time.perf_counter() - t0
    logging.info(f"Cycle time (match->snapshot->classify->log): {elapsed:.2f}s")

    # brief pause to avoid double-scans
    time.sleep(1.0)
    return True

def main():
    fp = init_fingerprint_sensor()
    logging.info("Waiting for finger... (Ctrl+C to quit)")
    while True:
        try:
            ran = process_one_cycle(fp)
            if not ran:
                time.sleep(CYCLE_SLEEP)
        except KeyboardInterrupt:
            print()
            logging.info("Exiting.")
            break
        except Exception as e:
            logging.error(f"Runtime error: {e}")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
