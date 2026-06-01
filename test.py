import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import time
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Initialize camera and models
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Increased width for better UI
cap.set(4, 1080)  # Increased height
detector = HandDetector(maxHands=2)  # Now supports two hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Enhanced UI Configuration
PRIMARY_COLOR = tuple(config["colors"]["primary"])
SECONDARY_COLOR = tuple(config["colors"]["secondary"])
BG_COLOR = tuple(config["colors"]["background"])
TEXT_COLOR = tuple(config["colors"]["text"])
HIGHLIGHT_COLOR = (0, 255, 255)  # Added for important info

# Load expanded labels
with open('Model/labels.txt') as f:
    labels = [line.split(' ', 1)[1].strip() for line in f.readlines()]

# System Variables
is_active = True
last_prediction = ""
confidence_threshold = 0.65  # Lowered for more gesture flexibility
history = []
two_handed_gestures = ["book", "computer", "good morning"]  # List of gestures needing two hands

def create_rounded_rect(img, rect, color, radius=25):  # Increased corner radius
    x, y, w, h = rect
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(img, (x + radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1)

def draw_scrollable_panel(img, title, items, position, size, max_items=8):  # New function
    x, y = position
    w, h = size
    create_rounded_rect(img, (x, y, w, h), (40, 40, 80))
    
    # Title
    cv2.putText(img, title, (x + 20, y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
    
    # Items (with scroll if more than max_items)
    start_idx = max(0, len(items) - max_items)
    for i, (gesture, timestamp) in enumerate(items[start_idx:]):
        cv2.putText(img, f"{i+1}. {gesture[:15]} ({timestamp})", (x + 20, y + 80 + i*40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_output = img.copy()
    
    # Create UI Canvas
    ui = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ui[:] = BG_COLOR
    
    # Main Camera View (Left Panel)
    camera_rect = (40, 40, 1200, 800)  # Larger area
    create_rounded_rect(ui, camera_rect, (30, 30, 60))
    if success:
        img_resized = cv2.resize(img, (1180, 780))  # Adjusted size
        ui[50:830, 50:1250] = img_resized
    
    # Info Panel (Right Panel)
    info_rect = (1300, 40, 580, 1000)  # Taller panel
    create_rounded_rect(ui, info_rect, (40, 40, 80))
    
    # Title with icon
    cv2.putText(ui, "SIGNVISION PRO", (1320, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, SECONDARY_COLOR, 3, cv2.LINE_AA)
    
    # Status Indicator
    status_color = SECONDARY_COLOR if is_active else (100, 100, 100)
    cv2.putText(ui, f"STATUS: {'ACTIVE' if is_active else 'PAUSED'}", (1320, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    
    # Gesture Recognition
    if is_active:
        hands, _ = detector.findHands(img_resized, flipType=False)
        
        if hands:
            # Process each hand (up to 2)
            for hand in hands[:2]:  # Only process first two hands
                x, y, w, h = hand['bbox']
                ui_x, ui_y = x + 50, y + 50
                
                # Process hand image
                img_crop = img_resized[y-30:y + h + 30, x-30:x + w + 30]  # Larger padding
                img_white = np.ones((400, 400, 3), np.uint8) * 255  # Larger canvas
                
                aspect_ratio = h / w
                
                try:
                    if aspect_ratio > 1:
                        k = 400 / h
                        w_cal = math.ceil(k * w)
                        img_resized_crop = cv2.resize(img_crop, (w_cal, 400))
                        w_gap = math.ceil((400 - w_cal) / 2)
                        img_white[:, w_gap:w_cal + w_gap] = img_resized_crop
                    else:
                        k = 400 / w
                        h_cal = math.ceil(k * h)
                        img_resized_crop = cv2.resize(img_crop, (400, h_cal))
                        h_gap = math.ceil((400 - h_cal) / 2)
                        img_white[h_gap:h_cal + h_gap, :] = img_resized_crop
                    
                    # Get prediction
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                    confidence = prediction[index] / 100
                    
                    if confidence > confidence_threshold:
                        current_gesture = labels[index]
                        
                        # Check if two hands are needed for this gesture
                        if current_gesture.lower() in two_handed_gestures and len(hands) < 2:
                            continue  # Skip if gesture requires two hands but only one detected
                            
                        last_prediction = current_gesture
                        history.append((last_prediction, time.strftime("%H:%M:%S")))
                        
                        # Visual feedback
                        cv2.rectangle(ui, (ui_x-30, ui_y-30), 
                                     (ui_x + w + 30, ui_y + h + 30), 
                                     HIGHLIGHT_COLOR if len(hands) == 2 else PRIMARY_COLOR, 
                                     3)
                        
                        # Label with confidence
                        label_bg = (ui_x-30, ui_y-80, w+60, 60)
                        create_rounded_rect(ui, label_bg, PRIMARY_COLOR)
                        cv2.putText(ui, f"{last_prediction.upper()} ({confidence:.0%})", 
                                   (ui_x-15, ui_y-40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)
                
                except Exception as e:
                    print(f"Processing error: {e}")
    
    # Display panels
    draw_scrollable_panel(ui, "CURRENT SIGN", 
                         [(last_prediction, time.strftime("%H:%M:%S"))] if last_prediction else [], 
                         (1320, 180), (540, 180))
    
    draw_scrollable_panel(ui, "RECENT SIGNS", history, (1320, 400), (540, 400))
    
    # Instructions
    cv2.putText(ui, "CONTROLS:", (1320, 850), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(ui, "SPACE: Toggle Recognition | Q: Quit", (1320, 890), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(ui, "Two-handed gestures supported:", (1320, 930), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, HIGHLIGHT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(ui, ", ".join(two_handed_gestures[:3]) + "...", (1320, 970), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    
    # Show UI
    cv2.imshow("SignVision Pro - ASL Recognition", ui)
    
    # Key Controls
    key = cv2.waitKey(1)
    if key == ord(' '):
        is_active = not is_active
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()