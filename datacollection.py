import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
is_recognition_active = True  # Start with recognition active

# 15 Gesture Mappings (expandable)
GESTURES = {
    (0,0,0,0,1): "Water",      # 'W' handshape (pinky extended)
    (1,1,1,1,1): "Help",       # Open hand wave
    (1,0,0,0,0): "Stop",       # Flat hand forward
    (0,1,0,0,0): "Bathroom",   # 'T' handshake
    (1,1,0,0,1): "I love you", # ILY handshape
    (1,0,0,0,1): "Hello",      # Salute motion
    (0,0,0,0,0): "No",         # Closed fist
    (1,1,1,1,1): "Yes",        # Nodding open hand
    (1,1,0,0,0): "Food",       # Pinched fingers to mouth
    (0,1,1,1,1): "Pain"        # Claw hand on pained area
}
def draw_button(img, text, position, size, color, text_color=(255, 255, 255)):
    x, y = position
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_AA)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    return (x, y, w, h)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Create UI background
    ui = np.zeros((img.shape[0], img.shape[1] + 300, 3), dtype=np.uint8)
    ui[:] = (40, 40, 40)
    ui[0:img.shape[0], 0:img.shape[1]] = img
    
    # Info panel
    info_panel = ui[0:img.shape[0], img.shape[1]:img.shape[1]+300]
    info_panel[:] = (30, 30, 30)
    
    # Draw buttons
    start_btn = draw_button(info_panel, "START", (50, 50), (200, 50), (0, 150, 0))
    stop_btn = draw_button(info_panel, "STOP", (50, 120), (200, 50), (0, 0, 150))
    
    # Display title
    cv2.putText(info_panel, "Gesture Recognition Pro", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    gesture = "No Hand Detected"
    fingers_state = ""
    
    if is_recognition_active:
        hands, _ = detector.findHands(img, draw=False)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            fingers = detector.fingersUp(hand)
            fingers_state = str(fingers)
            gesture = GESTURES.get(tuple(fingers), f"Unknown: {fingers}")
            
            # Draw bounding box only when active
            cv2.rectangle(ui, (x - offset, y - offset), 
                         (x + w + offset, y + h + offset), (0, 255, 0), 2)
            
            # Crop and resize hand image
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                cv2.imshow("Hand Crop", imgWhite)
    
    # Display status
    status_color = (0, 255, 0) if is_recognition_active else (0, 0, 255)
    status_text = "ACTIVE" if is_recognition_active else "INACTIVE"
    cv2.putText(info_panel, f"Status: {status_text}", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    cv2.putText(info_panel, f"Gesture: {gesture}", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(info_panel, f"Fingers: {fingers_state}", (50, 270), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_panel, f"Saved: {counter} images", (50, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
    
    # Display instructions
    cv2.putText(info_panel, "Press 's' to Save", (50, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_panel, "Press 'q' to Quit", (50, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Gesture Recognition Pro", ui)
    
    # Mouse click handler
    def mouse_callback(event, x, y, flags, param):
        global is_recognition_active
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if start button clicked
            if (start_btn[0] <= x - img.shape[1] <= start_btn[0] + start_btn[2] and 
                start_btn[1] <= y <= start_btn[1] + start_btn[3]):
                is_recognition_active = True
            # Check if stop button clicked
            elif (stop_btn[0] <= x - img.shape[1] <= stop_btn[0] + stop_btn[2] and 
                  stop_btn[1] <= y <= stop_btn[1] + stop_btn[3]):
                is_recognition_active = False
    
    cv2.setMouseCallback("Gesture Recognition Pro", mouse_callback)
    
    # Key controls
    key = cv2.waitKey(1)
    if key == ord("s"):
        if hands and is_recognition_active:
            counter += 1
            cv2.imwrite(f'Data/Collection/Image_{time.time()}.jpg', imgWhite)
            print(f"Saved: {gesture} (Total: {counter})")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()